from pathlib import Path

from FlagEmbedding import FlagModel

from proknow_rag.common.config import Settings
from proknow_rag.common.exceptions import EmbeddingError


class BGEM3Embedder:
    def __init__(self, settings: Settings | None = None):
        settings = settings or Settings()
        model_path = settings.bge_m3_model_path
        if not Path(model_path).exists():
            raise EmbeddingError(f"模型路径不存在: {model_path}，请先运行 scripts/download_models.py")
        self.model = FlagModel(model_path, use_fp16=True, devices="cuda")

    def embed(self, texts: list[str], batch_size: int = 12) -> dict:
        try:
            result = self.model.encode(
                texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
                batch_size=batch_size,
            )
            return {
                "dense_vectors": result["dense_vecs"],
                "sparse_vectors": result["lexical_weights"],
                "colbert_vectors": result["colbert_vecs"],
            }
        except Exception as e:
            raise EmbeddingError(f"嵌入计算失败: {e}") from e
