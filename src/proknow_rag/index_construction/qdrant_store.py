from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVectorParams,
    SparseVector,
    VectorParams,
    FusionQuery,
    Fusion,
)

from proknow_rag.common.config import Settings
from proknow_rag.common.exceptions import QdrantError


class QdrantEmbeddedStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        storage_path = str(Path(self.settings.qdrant_storage_path).resolve())
        try:
            self.client = QdrantClient(path=storage_path)
        except Exception as e:
            raise QdrantError(f"Qdrant 初始化失败: {e}") from e

    def ensure_collection(self, name: str = "proknow_rag") -> None:
        if self.collection_exists(name):
            return
        self.create_collection(name)

    def create_collection(self, name: str = "proknow_rag") -> None:
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

    def upsert(self, collection_name: str, points: list[dict]) -> None:
        try:
            qdrant_points = []
            for p in points:
                vector_data = p["vector"]
                sparse_data = vector_data.get("sparse")
                sparse_vector = None
                if sparse_data is not None:
                    if isinstance(sparse_data, SparseVector):
                        sparse_vector = sparse_data
                    elif isinstance(sparse_data, dict):
                        sparse_vector = SparseVector(
                            indices=sparse_data.get("indices", []),
                            values=sparse_data.get("values", []),
                        )
                    elif isinstance(sparse_data, (list, tuple)):
                        sparse_vector = SparseVector(
                            indices=list(range(len(sparse_data))),
                            values=[float(v) for v in sparse_data],
                        )

                named_vectors = {}
                if "dense" in vector_data:
                    named_vectors["dense"] = vector_data["dense"]
                if "colbert" in vector_data:
                    named_vectors["colbert"] = vector_data["colbert"]
                if sparse_vector is not None:
                    named_vectors["sparse"] = sparse_vector

                qdrant_points.append(
                    PointStruct(
                        id=p["id"],
                        vector=named_vectors,
                        payload=p.get("payload", {}),
                    )
                )
            self.client.upsert(collection_name=collection_name, points=qdrant_points, wait=True)
        except Exception as e:
            raise QdrantError(f"向量 upsert 失败: {e}") from e

    def search(
        self,
        collection_name: str,
        query_dense: list[float],
        query_sparse: SparseVector | None = None,
        query_colbert: list[list[float]] | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        filter_conditions: dict | None = None,
    ) -> list[dict]:
        try:
            prefetch_list = []

            prefetch_list.append(
                Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=limit * 3,
                )
            )

            if query_sparse is not None:
                prefetch_list.append(
                    Prefetch(
                        query=query_sparse,
                        using="sparse",
                        limit=limit * 3,
                    )
                )

            if query_colbert is not None:
                prefetch_list.append(
                    Prefetch(
                        query=query_colbert,
                        using="colbert",
                        limit=limit * 3,
                    )
                )

            results = self.client.query_points(
                collection_name=collection_name,
                prefetch=prefetch_list,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            )

            output = []
            for point in results.points:
                entry = {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload or {},
                }
                if score_threshold is not None and point.score < score_threshold:
                    continue
                output.append(entry)
            return output
        except Exception as e:
            raise QdrantError(f"向量检索失败: {e}") from e

    def delete(self, collection_name: str, point_ids: list[str]) -> None:
        try:
            from qdrant_client.models import PointIdsList

            self.client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=point_ids),
                wait=True,
            )
        except Exception as e:
            raise QdrantError(f"向量删除失败: {e}") from e

    def get_collection_info(self, collection_name: str) -> dict:
        try:
            info = self.client.get_collection(collection_name=collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": str(info.status),
                "optimizer_status": str(info.optimizer_status),
            }
        except Exception as e:
            raise QdrantError(f"获取 Collection 信息失败: {e}") from e
