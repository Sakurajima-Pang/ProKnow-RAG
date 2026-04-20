import hashlib
import uuid
from datetime import datetime, timezone

import structlog
from pydantic import BaseModel

from proknow_rag.common.config import Settings
from proknow_rag.common.exceptions import IndexConstructionError
from proknow_rag.data_preparation.base import PreparedChunk
from proknow_rag.index_construction.cache import EmbeddingCache
from proknow_rag.index_construction.embedder import BGEM3Embedder
from proknow_rag.index_construction.metadata_manager import MetadataManager
from proknow_rag.index_construction.qdrant_store import QdrantEmbeddedStore

logger = structlog.get_logger(__name__)


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
        self.cache = EmbeddingCache(self.settings)

    def _compute_chunk_hash(self, chunk: PreparedChunk) -> str:
        raw = f"{chunk.content}|{chunk.source}|{chunk.chunk_type}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _is_indexed(self, _collection_name: str, chunk_hash: str) -> bool:
        return self.cache.contains(chunk_hash)

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

        logger.info("开始嵌入计算", total=len(texts), batch_size=batch_size)
        try:
            embeddings = self.embedder.embed(texts, batch_size=batch_size)
        except Exception as e:
            raise IndexConstructionError(f"批量嵌入失败: {e}") from e

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
                processed_metadata["content"] = chunk.content

                dense_raw = embeddings["dense_vectors"][i]
                dense_vec = dense_raw.tolist() if hasattr(dense_raw, "tolist") else dense_raw

                sparse_dict = embeddings["sparse_vectors"][i]
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
                            "sparse": {"indices": sparse_indices, "values": sparse_values},
                        },
                        "payload": processed_metadata,
                    }
                )

                self.cache.put(chunk_hash)

            except Exception as e:
                failed_ids.append(chunk_hash)
                logger.warning("点构建失败", chunk_hash=chunk_hash, error=str(e))

        if points:
            try:
                self.store.upsert(collection_name, points)
            except Exception as e:
                raise IndexConstructionError(f"向量存储失败: {e}") from e

        for chunk_hash, _ in to_index:
            self.metadata_manager.set_version(
                chunk_hash,
                {"indexed": True, "collection": collection_name, "timestamp": datetime.now(timezone.utc).isoformat()},
            )

        return IndexingResult(
            collection_name=collection_name,
            num_indexed=len(points),
            failed_ids=failed_ids,
            index_version=self._generate_version(),
        )

    def _generate_version(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
