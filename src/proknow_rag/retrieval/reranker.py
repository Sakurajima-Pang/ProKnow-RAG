from pathlib import Path

from proknow_rag.common.config import Settings
from proknow_rag.common.exceptions import RerankerError
from proknow_rag.common.gpu_monitor import check_gpu_available


class BGEReranker:
    def __init__(self, model_path: str | None = None, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._model_path = model_path or self.settings.bge_reranker_model_path
        self._use_gpu = check_gpu_available(min_mb=2048)
        self.model = self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder

            model_name = self._model_path
            if self._use_gpu:
                try:
                    return CrossEncoder(model_name, device="cuda")
                except Exception:
                    self._use_gpu = False
                    return CrossEncoder(model_name, device="cpu")
            return CrossEncoder(model_name, device="cpu")
        except Exception as e:
            raise RerankerError(f"重排序模型加载失败: {e}") from e

    @property
    def use_gpu(self) -> bool:
        return self._use_gpu

    def _do_rerank(self, query: str, documents: list, batch_size: int = 32) -> list[tuple]:
        pairs = [(query, doc.content if hasattr(doc, "content") else str(doc)) for doc in documents]
        scores = self.model.predict(pairs, batch_size=batch_size)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return ranked

    def rerank(self, query: str, documents: list, top_k: int = 5) -> list:
        if not documents:
            return []
        try:
            ranked = self._do_rerank(query, documents, batch_size=32)
            return [doc for doc, _ in ranked[:top_k]]
        except Exception as e:
            if self._use_gpu:
                try:
                    self._use_gpu = False
                    self.model = self._load_model()
                    ranked = self._do_rerank(query, documents, batch_size=16)
                    return [doc for doc, _ in ranked[:top_k]]
                except Exception as fallback_err:
                    raise RerankerError(f"重排序失败(CPU降级后): {fallback_err}") from fallback_err
            raise RerankerError(f"重排序失败: {e}") from e

    def rerank_with_scores(self, query: str, documents: list, top_k: int = 5) -> list[tuple]:
        if not documents:
            return []
        try:
            ranked = self._do_rerank(query, documents, batch_size=32)
            return [(doc, float(score)) for doc, score in ranked[:top_k]]
        except Exception as e:
            if self._use_gpu:
                try:
                    self._use_gpu = False
                    self.model = self._load_model()
                    ranked = self._do_rerank(query, documents, batch_size=16)
                    return [(doc, float(score)) for doc, score in ranked[:top_k]]
                except Exception as fallback_err:
                    raise RerankerError(f"重排序失败(CPU降级后): {fallback_err}") from fallback_err
            raise RerankerError(f"重排序失败: {e}") from e
