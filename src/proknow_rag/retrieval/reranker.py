from sentence_transformers import CrossEncoder

from proknow_rag.common.exceptions import RerankerError


class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        try:
            self.model = CrossEncoder(model_name)
        except Exception as e:
            raise RerankerError(f"重排序模型加载失败: {e}") from e

    def rerank(self, query: str, documents: list, top_k: int = 5) -> list:
        if not documents:
            return []
        try:
            pairs = [(query, doc.content if hasattr(doc, "content") else str(doc)) for doc in documents]
            scores = self.model.predict(pairs, batch_size=32)
            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked[:top_k]]
        except Exception as e:
            raise RerankerError(f"重排序失败: {e}") from e
