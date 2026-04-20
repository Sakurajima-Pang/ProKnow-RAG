from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import structlog
from pydantic import BaseModel

from proknow_rag.common.config import Settings
from proknow_rag.common.exceptions import ProKnowRAGError
from proknow_rag.data_preparation.manager import DataManager
from proknow_rag.evaluation.metrics import LatencyStats, compute_latency_stats
from proknow_rag.evaluation.retrieval_eval import QueryEvaluation, RetrievalEvaluationResult, evaluate
from proknow_rag.index_construction.index_builder import IndexBuilder
from proknow_rag.retrieval.hybrid_search import HybridSearcher
from proknow_rag.retrieval.reranker import BGEReranker

logger = structlog.get_logger(__name__)


class IndexBuildStats(BaseModel):
    total_chunks: int
    build_time_sec: float
    chunks_per_sec: float


class QueryLatencyStats(BaseModel):
    search_latencies: LatencyStats
    rerank_latencies: LatencyStats
    end_to_end_latencies: LatencyStats


class BenchmarkResult(BaseModel):
    index_build: IndexBuildStats | None
    query_latency: QueryLatencyStats | None
    retrieval_quality: RetrievalEvaluationResult | None


class BenchmarkRunner:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()

    def benchmark_index_build(
        self,
        dir_path: str,
        collection_name: str = "proknow_rag",
        batch_size: int = 12,
    ) -> IndexBuildStats:
        data_manager = DataManager()
        index_builder = IndexBuilder(self.settings)

        start = time.perf_counter()
        chunks = data_manager.process_directory(dir_path)
        result = index_builder.build(chunks, collection_name=collection_name, batch_size=batch_size)
        elapsed = time.perf_counter() - start

        total = result.num_indexed
        return IndexBuildStats(
            total_chunks=total,
            build_time_sec=round(elapsed, 4),
            chunks_per_sec=round(total / elapsed, 2) if elapsed > 0 else 0.0,
        )

    def benchmark_query_latency(
        self,
        queries: Sequence[str],
        collection_name: str = "proknow_rag",
        top_k: int = 10,
        rerank_top_k: int = 5,
    ) -> QueryLatencyStats:
        searcher = HybridSearcher(self.settings)
        reranker = BGEReranker(settings=self.settings)

        search_latencies: list[float] = []
        rerank_latencies: list[float] = []
        e2e_latencies: list[float] = []

        for query in queries:
            e2e_start = time.perf_counter()

            search_start = time.perf_counter()
            results = searcher.search(query, collection_name=collection_name, limit=top_k)
            search_elapsed = time.perf_counter() - search_start
            search_latencies.append(search_elapsed)

            rerank_elapsed = 0.0
            if results:
                rerank_start = time.perf_counter()
                reranker.rerank(query, results, top_k=rerank_top_k)
                rerank_elapsed = time.perf_counter() - rerank_start
                rerank_latencies.append(rerank_elapsed)

            e2e_elapsed = time.perf_counter() - e2e_start
            e2e_latencies.append(e2e_elapsed)

        return QueryLatencyStats(
            search_latencies=compute_latency_stats(search_latencies),
            rerank_latencies=compute_latency_stats(rerank_latencies) if rerank_latencies else compute_latency_stats([]),
            end_to_end_latencies=compute_latency_stats(e2e_latencies),
        )

    def benchmark_retrieval_quality(
        self,
        queries: Sequence[str],
        relevant_ids_map: dict[str, set[str]],
        collection_name: str = "proknow_rag",
        top_k: int = 20,
        ks: Sequence[int] = (5, 10, 20),
    ) -> RetrievalEvaluationResult:
        searcher = HybridSearcher(self.settings)

        evaluations: list[QueryEvaluation] = []
        for query in queries:
            results = searcher.search(query, collection_name=collection_name, limit=top_k)
            retrieved = [r.id for r in results]
            relevant = relevant_ids_map.get(query, set())
            evaluations.append(QueryEvaluation(query=query, relevant_ids=relevant, retrieved_ids=retrieved))

        return evaluate(evaluations, ks=ks)

    def run_full_benchmark(
        self,
        dir_path: str | None = None,
        queries: Sequence[str] | None = None,
        relevant_ids_map: dict[str, set[str]] | None = None,
        collection_name: str = "proknow_rag",
        top_k: int = 20,
        rerank_top_k: int = 5,
        ks: Sequence[int] = (5, 10, 20),
    ) -> BenchmarkResult:
        index_stats = None
        if dir_path and Path(dir_path).exists():
            index_stats = self.benchmark_index_build(dir_path, collection_name=collection_name)

        query_stats = None
        if queries:
            query_stats = self.benchmark_query_latency(
                queries, collection_name=collection_name, top_k=top_k, rerank_top_k=rerank_top_k
            )

        quality_stats = None
        if queries and relevant_ids_map:
            quality_stats = self.benchmark_retrieval_quality(
                queries, relevant_ids_map, collection_name=collection_name, top_k=top_k, ks=ks
            )

        return BenchmarkResult(
            index_build=index_stats,
            query_latency=query_stats,
            retrieval_quality=quality_stats,
        )


def format_benchmark_report(result: BenchmarkResult) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("ProKnow-RAG Benchmark Report")
    lines.append("=" * 60)

    if result.index_build:
        ib = result.index_build
        lines.append("")
        lines.append("[Index Build]")
        lines.append(f"  Total Chunks:    {ib.total_chunks}")
        lines.append(f"  Build Time:      {ib.build_time_sec:.4f}s")
        lines.append(f"  Throughput:      {ib.chunks_per_sec:.2f} chunks/s")

    if result.query_latency:
        ql = result.query_latency
        lines.append("")
        lines.append("[Query Latency]")
        lines.append("  Search:")
        lines.append(f"    Mean:  {ql.search_latencies.mean * 1000:.2f}ms")
        lines.append(f"    P50:   {ql.search_latencies.p50 * 1000:.2f}ms")
        lines.append(f"    P95:   {ql.search_latencies.p95 * 1000:.2f}ms")
        lines.append(f"    P99:   {ql.search_latencies.p99 * 1000:.2f}ms")
        lines.append("  Rerank:")
        lines.append(f"    Mean:  {ql.rerank_latencies.mean * 1000:.2f}ms")
        lines.append(f"    P50:   {ql.rerank_latencies.p50 * 1000:.2f}ms")
        lines.append(f"    P95:   {ql.rerank_latencies.p95 * 1000:.2f}ms")
        lines.append(f"    P99:   {ql.rerank_latencies.p99 * 1000:.2f}ms")
        lines.append("  End-to-End:")
        lines.append(f"    Mean:  {ql.end_to_end_latencies.mean * 1000:.2f}ms")
        lines.append(f"    P50:   {ql.end_to_end_latencies.p50 * 1000:.2f}ms")
        lines.append(f"    P95:   {ql.end_to_end_latencies.p95 * 1000:.2f}ms")
        lines.append(f"    P99:   {ql.end_to_end_latencies.p99 * 1000:.2f}ms")

    if result.retrieval_quality:
        rq = result.retrieval_quality
        lines.append("")
        lines.append("[Retrieval Quality]")
        lines.append(f"  MRR:       {rq.mrr:.4f}")
        lines.append(f"  Hit Rate:  {rq.hit_rate:.4f}")
        for k, v in sorted(rq.ndcg_at_k.items()):
            lines.append(f"  NDCG@{k}:   {v:.4f}")
        for k, v in sorted(rq.recall_at_k.items()):
            lines.append(f"  Recall@{k}: {v:.4f}")
        for k, v in sorted(rq.precision_at_k.items()):
            lines.append(f"  Precision@{k}: {v:.4f}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
