import logging

from pathlib import Path

from FlagEmbedding import BGEM3FlagModel

from proknow_rag.common.config import Settings
from proknow_rag.common.exceptions import EmbeddingError
from proknow_rag.common.gpu_monitor import get_gpu_memory_info

logger = logging.getLogger(__name__)


class BGEM3Embedder:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        model_path = self.settings.bge_m3_model_path
        if not Path(model_path).exists():
            raise EmbeddingError(f"模型路径不存在: {model_path}，请先运行 scripts/download_models.py")

        self._use_gpu = self._detect_gpu()
        if self._use_gpu:
            self.model = BGEM3FlagModel(
                model_path,
                use_fp16=True,
                devices="cuda",
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
            )
            self.model.model.to("cuda")
            self.model.model.eval()
            logger.info("BGE-M3 loaded on GPU (fp16)")
        else:
            self.model = BGEM3FlagModel(
                model_path,
                use_fp16=False,
                devices="cpu",
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
            )
            logger.info("BGE-M3 loaded on CPU (fp32)")

    @staticmethod
    def _detect_gpu() -> bool:
        gpu_info = get_gpu_memory_info()
        if gpu_info["total_mb"] == 0:
            return False
        if gpu_info["free_mb"] < 3000:
            logger.warning(f"GPU free memory too low ({gpu_info['free_mb']}MB), falling back to CPU")
            return False
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            _ = torch.cuda.current_device()
            return True
        except Exception:
            return False

    @property
    def use_gpu(self) -> bool:
        return self._use_gpu

    def _compute_batch_size(self, requested_batch_size: int) -> int:
        if not self._use_gpu:
            return min(requested_batch_size, 4)
        gpu_info = get_gpu_memory_info()
        free_mb = gpu_info.get("free_mb", 0)
        total_mb = gpu_info.get("total_mb", 1)
        usage_ratio = 1.0 - (free_mb / total_mb)
        if usage_ratio > 0.85:
            return min(requested_batch_size, 2)
        elif usage_ratio > 0.7:
            return min(requested_batch_size, 4)
        elif usage_ratio > 0.5:
            return min(requested_batch_size, 8)
        else:
            return min(requested_batch_size, 12)

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
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self._use_gpu:
                logger.warning("GPU OOM, reducing batch size and retrying")
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    result = self.model.encode(
                        texts,
                        return_dense=True,
                        return_sparse=True,
                        return_colbert_vecs=True,
                        batch_size=max(1, effective_batch_size // 4),
                    )
                    return {
                        "dense_vectors": result["dense_vecs"],
                        "sparse_vectors": result["lexical_weights"],
                        "colbert_vectors": result["colbert_vecs"],
                    }
                except Exception as retry_err:
                    raise EmbeddingError(f"GPU OOM 后重试失败: {retry_err}") from retry_err
            raise EmbeddingError(f"嵌入计算失败: {e}") from e
        except Exception as e:
            raise EmbeddingError(f"嵌入计算失败: {e}") from e
