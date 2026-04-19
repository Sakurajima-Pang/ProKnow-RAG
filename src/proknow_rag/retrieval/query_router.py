import re

from pydantic import BaseModel

from proknow_rag.retrieval.hybrid_search import WEIGHT_PRESETS


class RetrievalStrategy(BaseModel):
    dense_weight: float = 0.5
    sparse_weight: float = 0.3
    colbert_weight: float = 0.2
    bm25_weight: float = 0.0
    doc_type_filter: str | None = None


STRATEGY_PRESETS = {
    "paper": RetrievalStrategy(dense_weight=0.6, sparse_weight=0.3, colbert_weight=0.1, bm25_weight=0.0, doc_type_filter="paper"),
    "doc": RetrievalStrategy(dense_weight=0.5, sparse_weight=0.3, colbert_weight=0.2, bm25_weight=0.0, doc_type_filter="doc"),
    "code": RetrievalStrategy(dense_weight=0.3, sparse_weight=0.4, colbert_weight=0.3, bm25_weight=0.0, doc_type_filter="code"),
    "general": RetrievalStrategy(dense_weight=0.5, sparse_weight=0.3, colbert_weight=0.2, bm25_weight=0.0),
}

CODE_KEYWORDS = [
    "函数", "function", "实现", "implement", "代码", "code", "类", "class",
    "方法", "method", "api", "接口", "import", "def", "return", "变量",
    "variable", "模块", "module", "包", "package", "库", "library",
    "调试", "debug", "编译", "compile", "运行", "run", "部署", "deploy",
    "参数", "parameter", "argument", "回调", "callback", "继承", "inherit",
    "封装", "encapsulate", "多态", "polymorphism",
]

PAPER_KEYWORDS = [
    "论文", "paper", "研究", "research", "实验", "experiment", "算法", "algorithm",
    "模型", "model", "精度", "accuracy", "基准", "benchmark", "数据集", "dataset",
    "训练", "training", "推理", "inference", "微调", "finetune", "预训练", "pretrain",
    "损失函数", "loss function", "优化", "optimization", "收敛", "convergence",
    "消融", "ablation", "SOTA", "state-of-the-art",
]

ZH_PATTERN = re.compile(r"[\u4e00-\u9fff]")
EN_PATTERN = re.compile(r"[a-zA-Z]")


class QueryRouter:
    def _detect_language(self, query: str) -> str:
        zh_count = len(ZH_PATTERN.findall(query))
        en_count = len(EN_PATTERN.findall(query))
        if zh_count > en_count:
            return "zh"
        elif en_count > zh_count:
            return "en"
        return "mixed"

    def _compute_query_length_category(self, query: str) -> str:
        length = len(query.strip())
        if length <= 10:
            return "short"
        elif length <= 50:
            return "medium"
        return "long"

    def _count_keyword_matches(self, query_lower: str, keywords: list[str]) -> int:
        return sum(1 for kw in keywords if kw in query_lower)

    def route(self, query: str) -> RetrievalStrategy:
        query_lower = query.lower()
        code_score = self._count_keyword_matches(query_lower, CODE_KEYWORDS)
        paper_score = self._count_keyword_matches(query_lower, PAPER_KEYWORDS)
        length_category = self._compute_query_length_category(query)
        language = self._detect_language(query)

        if code_score > paper_score and code_score >= 2:
            strategy = STRATEGY_PRESETS["code"].model_copy()
        elif paper_score > code_score and paper_score >= 2:
            strategy = STRATEGY_PRESETS["paper"].model_copy()
        elif code_score == paper_score and code_score > 0:
            strategy = STRATEGY_PRESETS["general"].model_copy()
        else:
            strategy = STRATEGY_PRESETS["general"].model_copy()

        if length_category == "short":
            strategy.sparse_weight = min(strategy.sparse_weight + 0.1, 0.5)
            strategy.dense_weight = max(strategy.dense_weight - 0.1, 0.2)
        elif length_category == "long":
            strategy.dense_weight = min(strategy.dense_weight + 0.1, 0.7)
            strategy.sparse_weight = max(strategy.sparse_weight - 0.1, 0.2)

        if language == "zh":
            strategy.sparse_weight = min(strategy.sparse_weight + 0.05, 0.5)
        elif language == "en":
            strategy.colbert_weight = min(strategy.colbert_weight + 0.05, 0.4)

        total = strategy.dense_weight + strategy.sparse_weight + strategy.colbert_weight + strategy.bm25_weight
        if total > 0:
            strategy.dense_weight /= total
            strategy.sparse_weight /= total
            strategy.colbert_weight /= total
            strategy.bm25_weight /= total

        return strategy

    def route_to_weights(self, query: str) -> dict[str, float]:
        strategy = self.route(query)
        return {
            "dense": strategy.dense_weight,
            "sparse": strategy.sparse_weight,
            "colbert": strategy.colbert_weight,
            "bm25": strategy.bm25_weight,
        }
