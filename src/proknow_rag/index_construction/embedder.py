from pathlib import Path

from FlagEmbedding import BGEM3FlagModel

from proknow_rag.common.config import Settings
from proknow_rag.common.exceptions import EmbeddingError
from proknow_rag.common.gpu_monitor import check_gpu_available, get_gpu_memory_info


class BGEM3Embedder:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        model_path = self.settings.bge_m3_model_path
        if not Path(model_path).exists():
            raise EmbeddingError(f"模型路径不存在: {model_path}，请先运行 scripts/download_models.py")
        self._use_gpu = check_gpu_available(min_mb=2048)
        if self._use_gpu:
            try:
                self.model = BGEM3FlagModel(
                    model_path,
                    use_fp16=True,
                    devices="cuda",
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=True,
                )
            except Exception:
                self._use_gpu = False
                self.model = BGEM3FlagModel(
                    model_path,
                    use_fp16=False,
                    devices="cpu",
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=True,
                )
        else:
            self.model = BGEM3FlagModel(
                model_path,
                use_fp16=False,
                devices="cpu",
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
            )

    @property
    def use_gpu(self) -> bool:
        return self._use_gpu

    def _compute_batch_size(self, requested_batch_size: int) -> int:
        if not self._use_gpu:
            return min(requested_batch_size, 4)
        gpu_info = get_gpu_memory_info()
        free_mb = gpu_info.get("free_mb", 0)
        if free_mb >= 20000:
            return min(requested_batch_size, 32)
        elif free_mb >= 12000:
            return min(requested_batch_size, 16)
        elif free_mb >= 6000:
            return min(requested_batch_size, 8)
        else:
            return min(requested_batch_size, 4)

    def embed(self, texts: list[str], batch_size: int = 12) -> dict:
        if not texts:
            return {"dense_vectors": [], "sparse_vectors": [], "colbert_vectors": []}
        effective_batch_size = self._compute_batch_size(batch_size)
        try:
            result = self.model.encode(
                texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
                batch_size=effective_batch_size,
            )
            return {
                "dense_vectors": result["dense_vecs"],
                "sparse_vectors": result["lexical_weights"],
                "colbert_vectors": result["colbert_vecs"],
            }
        except Exception as e:
            if self._use_gpu:
                try:
                    self._use_gpu = False
                    model_path = self.settings.bge_m3_model_path
                    self.model = BGEM3FlagModel(
                        model_path,
                        use_fp16=False,
                        devices="cpu",
                        return_dense=True,
                        return_sparse=True,
                        return_colbert_vecs=True,
                    )
                    result = self.model.encode(
                        texts,
                        return_dense=True,
                        return_sparse=True,
                        return_colbert_vecs=True,
                        batch_size=min(batch_size, 4),
                    )
                    return {
                        "dense_vectors": result["dense_vecs"],
                        "sparse_vectors": result["lexical_weights"],
                        "colbert_vectors": result["colbert_vecs"],
                    }
                except Exception as fallback_err:
                    raise EmbeddingError(f"嵌入计算失败(GPU降级CPU后): {fallback_err}") from fallback_err
            raise EmbeddingError(f"嵌入计算失败: {e}") from e
