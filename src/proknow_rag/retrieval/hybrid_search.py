import re
from dataclasses import dataclass, field

from qdrant_client.models import SparseVector

from proknow_rag.common.config import Settings
from proknow_rag.common.exceptions import RetrievalError
from proknow_rag.index_construction.embedder import BGEM3Embedder
from proknow_rag.index_construction.qdrant_store import QdrantEmbeddedStore


@dataclass
class SearchResult:
    id: str
    score: float
    payload: dict = field(default_factory=dict)
    content: str = ""


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "paper": {"dense": 0.6, "sparse": 0.3, "colbert": 0.1, "bm25": 0.0},
    "doc": {"dense": 0.5, "sparse": 0.3, "colbert": 0.2, "bm25": 0.0},
    "code": {"dense": 0.3, "sparse": 0.4, "colbert": 0.3, "bm25": 0.0},
    "general": {"dense": 0.5, "sparse": 0.3, "colbert": 0.2, "bm25": 0.0},
}


class HybridSearcher:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.embedder = BGEM3Embedder(self.settings)
        self.store = QdrantEmbeddedStore(self.settings)
        self._bm25_corpus: list[str] = []
        self._bm25_model = None
        self._bm25_doc_ids: list[str] = []

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _build_sparse_vector(self, sparse_dict: dict) -> SparseVector:
        indices = [int(k) for k in sparse_dict.keys()]
        values = [float(v) for v in sparse_dict.values()]
        return SparseVector(indices=indices, values=values)

    def _dense_search(self, collection_name: str, query_dense: list[float], limit: int) -> list[SearchResult]:
        try:
            from qdrant_client.models import Prefetch, FusionQuery, Fusion

            results = self.store.client.query_points(
                collection_name=collection_name,
                prefetch=[Prefetch(query=query_dense, using="dense", limit=limit)],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
            return [
                SearchResult(
                    id=str(p.id),
                    score=p.score,
                    payload=p.payload or {},
                    content=p.payload.get("content", "") if p.payload else "",
                )
                for p in results.points
            ]
        except Exception as e:
            raise RetrievalError(f"Dense 检索失败: {e}") from e

    def _sparse_search(self, collection_name: str, query_sparse: SparseVector, limit: int) -> list[SearchResult]:
        try:
            from qdrant_client.models import Prefetch, FusionQuery, Fusion

            results = self.store.client.query_points(
                collection_name=collection_name,
                prefetch=[Prefetch(query=query_sparse, using="sparse", limit=limit)],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
            return [
                SearchResult(
                    id=str(p.id),
                    score=p.score,
                    payload=p.payload or {},
                    content=p.payload.get("content", "") if p.payload else "",
                )
                for p in results.points
            ]
        except Exception as e:
            raise RetrievalError(f"Sparse 检索失败: {e}") from e

    def _colbert_search(self, collection_name: str, query_colbert: list[list[float]], limit: int) -> list[SearchResult]:
        try:
            from qdrant_client.models import Prefetch, FusionQuery, Fusion

            results = self.store.client.query_points(
                collection_name=collection_name,
                prefetch=[Prefetch(query=query_colbert, using="colbert", limit=limit)],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
            return [
                SearchResult(
                    id=str(p.id),
                    score=p.score,
                    payload=p.payload or {},
                    content=p.payload.get("content", "") if p.payload else "",
                )
                for p in results.points
            ]
        except Exception as e:
            raise RetrievalError(f"ColBERT 检索失败: {e}") from e

    def _bm25_search(self, query: str, limit: int) -> list[SearchResult]:
        if self._bm25_model is None:
            return []
        try:
            tokenized_query = self._tokenize(query)
            scores = self._bm25_model.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
            results = []
            for idx in top_indices:
                if scores[idx] > 0 and idx < len(self._bm25_doc_ids):
                    results.append(
                        SearchResult(
                            id=self._bm25_doc_ids[idx],
                            score=float(scores[idx]),
                            content=self._bm25_corpus[idx] if idx < len(self._bm25_corpus) else "",
                        )
                    )
            return results
        except Exception as e:
            raise RetrievalError(f"BM25 检索失败: {e}") from e

    def build_bm25_index(self, documents: list[dict]) -> None:
        try:
            from rank_bm25 import BM25Okapi

            self._bm25_corpus = []
            self._bm25_doc_ids = []
            tokenized_corpus = []

            for doc in documents:
                content = doc.get("content", "")
                doc_id = doc.get("id", "")
                self._bm25_corpus.append(content)
                self._bm25_doc_ids.append(doc_id)
                tokenized_corpus.append(self._tokenize(content))

            self._bm25_model = BM25Okapi(tokenized_corpus)
        except Exception as e:
            raise RetrievalError(f"BM25 索引构建失败: {e}") from e

    def search(
        self,
        query: str,
        collection_name: str = "proknow_rag",
        limit: int = 10,
        weights: dict[str, float] | None = None,
        doc_type: str | None = None,
    ) -> list[SearchResult]:
        try:
            if weights is None:
                preset_key = doc_type if doc_type and doc_type in WEIGHT_PRESETS else "general"
                weights = WEIGHT_PRESETS[preset_key]

            embeddings = self.embedder.embed([query], batch_size=1)
            query_dense = embeddings["dense_vectors"][0].tolist() if hasattr(embeddings["dense_vectors"][0], "tolist") else embeddings["dense_vectors"][0]
            sparse_dict = embeddings["sparse_vectors"][0]
            query_sparse = self._build_sparse_vector(sparse_dict)
            colbert_raw = embeddings["colbert_vectors"][0]
            query_colbert = (
                [v.tolist() if hasattr(v, "tolist") else v for v in colbert_raw]
                if isinstance(colbert_raw, list)
                else colbert_raw.tolist() if hasattr(colbert_raw, "tolist") else colbert_raw
            )

            all_results: dict[str, list[tuple[int, float]]] = {}
            doc_data: dict[str, SearchResult] = {}

            search_limit = limit * 3

            if weights.get("dense", 0) > 0:
                dense_results = self._dense_search(collection_name, query_dense, search_limit)
                for rank, r in enumerate(dense_results):
                    all_results.setdefault(r.id, []).append((rank, weights["dense"]))
                    if r.id not in doc_data:
                        doc_data[r.id] = r

            if weights.get("sparse", 0) > 0:
                sparse_results = self._sparse_search(collection_name, query_sparse, search_limit)
                for rank, r in enumerate(sparse_results):
                    all_results.setdefault(r.id, []).append((rank, weights["sparse"]))
                    if r.id not in doc_data:
                        doc_data[r.id] = r

            if weights.get("colbert", 0) > 0:
                colbert_results = self._colbert_search(collection_name, query_colbert, search_limit)
                for rank, r in enumerate(colbert_results):
                    all_results.setdefault(r.id, []).append((rank, weights["colbert"]))
                    if r.id not in doc_data:
                        doc_data[r.id] = r

            if weights.get("bm25", 0) > 0:
                bm25_results = self._bm25_search(query, search_limit)
                for rank, r in enumerate(bm25_results):
                    all_results.setdefault(r.id, []).append((rank, weights["bm25"]))
                    if r.id not in doc_data:
                        doc_data[r.id] = r

            fused_scores: dict[str, float] = {}
            k = 60
            for doc_id, rank_weight_pairs in all_results.items():
                score = 0.0
                for rank, weight in rank_weight_pairs:
                    score += weight / (k + rank + 1)
                fused_scores[doc_id] = score

            sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            results = []
            for doc_id, score in sorted_ids[:limit]:
                result = doc_data.get(doc_id, SearchResult(id=doc_id, score=score))
                result.score = score
                results.append(result)
            return results
        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(f"混合检索失败: {e}") from e
