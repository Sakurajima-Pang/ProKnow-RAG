from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from proknow_rag.common.config import Settings
from proknow_rag.common.exceptions import QdrantError


class QdrantEmbeddedStore:
    def __init__(self, settings: Settings | None = None):
        settings = settings or Settings()
        storage_path = str(Path(settings.qdrant_storage_path).resolve())
        try:
            self.client = QdrantClient(path=storage_path)
        except Exception as e:
            raise QdrantError(f"Qdrant 初始化失败: {e}") from e

    def create_collection(self, name: str = "proknow_rag"):
        try:
            self.client.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": VectorParams(size=1024, distance=Distance.COSINE),
                    "colbert": VectorParams(
                        size=128,
                        distance=Distance.COSINE,
                        multivector_config=MultiVectorConfig(
                            comparator=MultiVectorComparator.MAX_SIM
                        ),
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
                },
            )
        except Exception as e:
            raise QdrantError(f"Collection 创建失败: {e}") from e

    def collection_exists(self, name: str = "proknow_rag") -> bool:
        return any(c.name == name for c in self.client.get_collections().collections)
