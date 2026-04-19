from pydantic import BaseModel


class RetrievalStrategy(BaseModel):
    dense_weight: float = 0.5
    sparse_weight: float = 0.3
    colbert_weight: float = 0.2
    doc_type_filter: str | None = None


STRATEGY_PRESETS = {
    "paper": RetrievalStrategy(dense_weight=0.6, sparse_weight=0.3, colbert_weight=0.1, doc_type_filter="paper"),
    "doc": RetrievalStrategy(dense_weight=0.5, sparse_weight=0.3, colbert_weight=0.2, doc_type_filter="doc"),
    "code": RetrievalStrategy(dense_weight=0.3, sparse_weight=0.4, colbert_weight=0.3, doc_type_filter="code"),
    "general": RetrievalStrategy(dense_weight=0.5, sparse_weight=0.3, colbert_weight=0.2),
}


class QueryRouter:
    def route(self, query: str) -> RetrievalStrategy:
        query_lower = query.lower()
        code_keywords = ["函数", "function", "实现", "implement", "代码", "code", "类", "class", "方法", "method", "api", "接口"]
        paper_keywords = ["论文", "paper", "研究", "research", "实验", "experiment", "算法", "algorithm", "模型", "model"]

        if any(kw in query_lower for kw in code_keywords):
            return STRATEGY_PRESETS["code"]
        elif any(kw in query_lower for kw in paper_keywords):
            return STRATEGY_PRESETS["paper"]
        return STRATEGY_PRESETS["general"]
