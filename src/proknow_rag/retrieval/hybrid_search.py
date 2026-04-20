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


WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "paper": {"dense": 0.65, "sparse": 0.35, "bm25": 0.0},
    "doc": {"dense": 0.6, "sparse": 0.4, "bm25": 0.0},
    "code": {"dense": 0.4, "sparse": 0.6, "bm25": 0.0},
    "general": {"dense": 0.6, "sparse": 0.4, "bm25": 0.0},
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

            use_dense = weights.get("dense", 0) > 0
            use_sparse = weights.get("sparse", 0) > 0

            if use_dense and use_sparse:
                raw_results = self.store.search(
                    collection_name=collection_name,
                    query_dense=query_dense,
                    query_sparse=query_sparse,
                    limit=limit,
                )
            elif use_dense:
                raw_results = self.store.search(
                    collection_name=collection_name,
                    query_dense=query_dense,
                    query_sparse=None,
                    limit=limit,
                )
            elif use_sparse:
                raw_results = self._sparse_only_search(collection_name, query_sparse, limit)
            else:
                raw_results = []

            results = []
            for r in raw_results:
                payload = r.get("payload", {})
                results.append(
                    SearchResult(
                        id=r["id"],
                        score=r["score"],
                        payload=payload,
                        content=payload.get("content", ""),
                    )
                )

            if weights.get("bm25", 0) > 0 and self._bm25_model is not None:
                bm25_results = self._bm25_search(query, limit)
                results = self._rrf_fuse(results, bm25_results, weights)

            return results[:limit]
        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(f"混合检索失败: {e}") from e

    def _sparse_only_search(self, collection_name: str, query_sparse: SparseVector, limit: int) -> list[dict]:
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
                {
                    "id": str(p.id),
                    "score": p.score,
                    "payload": p.payload or {},
                }
                for p in results.points
            ]
        except Exception as e:
            raise RetrievalError(f"Sparse 检索失败: {e}") from e

    def _rrf_fuse(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        weights: dict[str, float],
        k: int = 60,
    ) -> list[SearchResult]:
        scores: dict[str, float] = {}
        doc_data: dict[str, SearchResult] = {}

        vector_weight = weights.get("dense", 0.5) + weights.get("sparse", 0.5)
        bm25_weight = weights.get("bm25", 0.0)

        for rank, r in enumerate(vector_results):
            scores[r.id] = scores.get(r.id, 0) + vector_weight / (k + rank + 1)
            if r.id not in doc_data:
                doc_data[r.id] = r

        for rank, r in enumerate(bm25_results):
            scores[r.id] = scores.get(r.id, 0) + bm25_weight / (k + rank + 1)
            if r.id not in doc_data:
                doc_data[r.id] = r

        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in sorted_ids:
            result = doc_data.get(doc_id, SearchResult(id=doc_id, score=score))
            result.score = score
            results.append(result)
        return results
