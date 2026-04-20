import threading
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointIdsList,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVectorParams,
    SparseVector,
    VectorParams,
    FusionQuery,
    Fusion,
    Filter,
    FieldCondition,
    MatchValue,
)

from proknow_rag.common.config import Settings
from proknow_rag.common.exceptions import QdrantError

_client_lock = threading.Lock()
_client_instance: QdrantClient | None = None
_client_storage_path: str | None = None


def _get_shared_client(storage_path: str) -> QdrantClient:
    global _client_instance, _client_storage_path
    with _client_lock:
        if _client_instance is not None and _client_storage_path == storage_path:
            return _client_instance
        if _client_instance is not None:
            try:
                _client_instance.close()
            except Exception:
                pass
            _client_instance = None
        _client_instance = QdrantClient(path=storage_path)
        _client_storage_path = storage_path
        return _client_instance


class QdrantEmbeddedStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        storage_path = str(Path(self.settings.qdrant_storage_path).resolve())
        try:
            self.client = _get_shared_client(storage_path)
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

    def _build_filter(self, filter_conditions: dict | None) -> Filter | None:
        if not filter_conditions:
            return None
        conditions = []
        for field, value in filter_conditions.items():
            conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
        return Filter(must=conditions) if conditions else None

    def search(
        self,
        collection_name: str,
        query_dense: list[float],
        query_sparse: SparseVector | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        filter_conditions: dict | None = None,
    ) -> list[dict]:
        try:
            qdrant_filter = self._build_filter(filter_conditions)
            prefetch_list = []

            prefetch_list.append(
                Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=limit * 3,
                    filter=qdrant_filter,
                )
            )

            if query_sparse is not None:
                prefetch_list.append(
                    Prefetch(
                        query=query_sparse,
                        using="sparse",
                        limit=limit * 3,
                        filter=qdrant_filter,
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
                "vectors_count": info.indexed_vectors_count or 0,
                "points_count": info.points_count or 0,
                "status": str(info.status),
                "optimizer_status": str(info.optimizer_status),
            }
        except Exception as e:
            raise QdrantError(f"获取 Collection 信息失败: {e}") from e
