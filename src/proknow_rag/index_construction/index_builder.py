import hashlib
import uuid
from datetime import datetime

from pydantic import BaseModel

from proknow_rag.common.config import Settings
from proknow_rag.common.exceptions import IndexConstructionError
from proknow_rag.data_preparation.base import PreparedChunk
from proknow_rag.index_construction.cache import EmbeddingCache
from proknow_rag.index_construction.embedder import BGEM3Embedder
from proknow_rag.index_construction.metadata_manager import MetadataManager
from proknow_rag.index_construction.qdrant_store import QdrantEmbeddedStore


class IndexingResult(BaseModel):
    collection_name: str
    num_indexed: int
    failed_ids: list[str]
    index_version: str


class IndexBuilder:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.embedder = BGEM3Embedder(self.settings)
        self.store = QdrantEmbeddedStore(self.settings)
        self.metadata_manager = MetadataManager()
        self.cache = EmbeddingCache()

    def _compute_chunk_hash(self, chunk: PreparedChunk) -> str:
        raw = f"{chunk.content}|{chunk.source}|{chunk.chunk_type}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _is_indexed(self, collection_name: str, chunk_hash: str) -> bool:
        record = self.metadata_manager.get_version(chunk_hash)
        return record is not None and record.get("indexed", False)

    def build(
        self,
        chunks: list[PreparedChunk],
        collection_name: str = "proknow_rag",
        incremental: bool = True,
        batch_size: int = 12,
    ) -> IndexingResult:
        self.store.ensure_collection(collection_name)

        to_index: list[tuple[str, PreparedChunk]] = []
        for chunk in chunks:
            chunk_hash = self._compute_chunk_hash(chunk)
            if incremental and self._is_indexed(collection_name, chunk_hash):
                continue
            to_index.append((chunk_hash, chunk))

        if not to_index:
            return IndexingResult(
                collection_name=collection_name,
                num_indexed=0,
                failed_ids=[],
                index_version=self._generate_version(),
            )

        texts = [chunk.content for _, chunk in to_index]
        chunk_hashes = [h for h, _ in to_index]

        cached_results = {}
        uncached_indices = []
        uncached_texts = []

        for i, (chunk_hash, text) in enumerate(zip(chunk_hashes, texts)):
            cached = self.cache.get(chunk_hash)
            if cached is not None:
                cached_results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        new_embeddings = {}
        if uncached_texts:
            try:
                new_embeddings = self.embedder.embed(uncached_texts, batch_size=batch_size)
            except Exception as e:
                raise IndexConstructionError(f"批量嵌入失败: {e}") from e

        all_dense = [None] * len(texts)
        all_sparse = [None] * len(texts)
        all_colbert = [None] * len(texts)

        for i, cached in cached_results.items():
            all_dense[i] = cached["dense_vectors"]
            all_sparse[i] = cached["sparse_vectors"]
            all_colbert[i] = cached["colbert_vectors"]

        for idx, orig_i in enumerate(uncached_indices):
            dense = new_embeddings["dense_vectors"][idx]
            sparse = new_embeddings["sparse_vectors"][idx]
            colbert = new_embeddings["colbert_vectors"][idx]
            all_dense[orig_i] = dense
            all_sparse[orig_i] = sparse
            all_colbert[orig_i] = colbert

            chunk_hash = chunk_hashes[orig_i]
            self.cache.put(
                chunk_hash,
                {
                    "dense_vectors": dense,
                    "sparse_vectors": sparse,
                    "colbert_vectors": colbert,
                },
            )

        points = []
        failed_ids = []
        for i, (chunk_hash, chunk) in enumerate(to_index):
            try:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_hash))
                processed_metadata = self.metadata_manager.process_metadata(chunk.metadata)
                processed_metadata["doc_hash"] = chunk.doc_hash or chunk_hash
                processed_metadata["chunk_hash"] = chunk_hash
                processed_metadata["source"] = chunk.source
                processed_metadata["chunk_type"] = chunk.chunk_type

                dense_vec = all_dense[i].tolist() if hasattr(all_dense[i], "tolist") else all_dense[i]
                colbert_vec = (
                    [v.tolist() if hasattr(v, "tolist") else v for v in all_colbert[i]]
                    if isinstance(all_colbert[i], list)
                    else all_colbert[i].tolist() if hasattr(all_colbert[i], "tolist") else all_colbert[i]
                )
                sparse_dict = all_sparse[i]
                sparse_indices = []
                sparse_values = []
                if isinstance(sparse_dict, dict):
                    sparse_indices = [int(k) for k in sparse_dict.keys()]
                    sparse_values = [float(v) for v in sparse_dict.values()]

                points.append(
                    {
                        "id": point_id,
                        "vector": {
                            "dense": dense_vec,
                            "colbert": colbert_vec,
                            "sparse": {"indices": sparse_indices, "values": sparse_values},
                        },
                        "payload": processed_metadata,
                    }
                )
            except Exception:
                failed_ids.append(chunk_hash)

        if points:
            try:
                self.store.upsert(collection_name, points)
            except Exception as e:
                raise IndexConstructionError(f"向量存储失败: {e}") from e

        for chunk_hash, _ in to_index:
            self.metadata_manager.set_version(
                chunk_hash,
                {"indexed": True, "collection": collection_name, "timestamp": datetime.utcnow().isoformat()},
            )

        return IndexingResult(
            collection_name=collection_name,
            num_indexed=len(points),
            failed_ids=failed_ids,
            index_version=self._generate_version(),
        )

    def _generate_version(self) -> str:
        return datetime.utcnow().strftime("%Y%m%d%H%M%S")
