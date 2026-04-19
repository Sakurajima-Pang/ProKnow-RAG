import re

from proknow_rag.common.exceptions import RetrievalError

TECH_SYNONYMS: dict[str, list[str]] = {
    "k8s": ["kubernetes"],
    "kubernetes": ["k8s"],
    "ml": ["machine learning", "机器学习"],
    "dl": ["deep learning", "深度学习"],
    "nlp": ["natural language processing", "自然语言处理"],
    "llm": ["large language model", "大语言模型"],
    "rag": ["retrieval augmented generation", "检索增强生成"],
    "rnn": ["recurrent neural network", "循环神经网络"],
    "cnn": ["convolutional neural network", "卷积神经网络"],
    "gpt": ["generative pre-trained transformer"],
    "bert": ["bidirectional encoder representations from transformers"],
    "api": ["application programming interface", "应用程序接口"],
    "sql": ["structured query language"],
    "db": ["database", "数据库"],
    "os": ["operating system", "操作系统"],
    "ci": ["continuous integration", "持续集成"],
    "cd": ["continuous deployment", "持续部署", "continuous delivery", "持续交付"],
    "devops": ["development operations"],
    "sre": ["site reliability engineering"],
    "vm": ["virtual machine", "虚拟机"],
    "cpu": ["central processing unit", "中央处理器"],
    "gpu": ["graphics processing unit", "图形处理器"],
    "tpu": ["tensor processing unit"],
    "rdbms": ["relational database management system", "关系型数据库"],
    "orm": ["object-relational mapping", "对象关系映射"],
    "rest": ["representational state transfer"],
    "rpc": ["remote procedure call", "远程过程调用"],
    "grpc": ["google remote procedure call"],
    "etl": ["extract transform load"],
    "olap": ["online analytical processing"],
    "oltp": ["online transaction processing"],
}

STOP_WORDS_EN: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "about",
    "please", "help", "want", "tell", "show", "give", "know",
}

STOP_WORDS_ZH: set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会",
    "着", "没有", "看", "好", "自己", "这", "他", "她", "它", "们",
    "那", "些", "什么", "怎么", "如何", "哪", "为什么", "请问", "能",
    "可以", "吗", "吧", "啊", "呢", "嗯", "哦", "呀", "哈", "嘿",
}


class QueryRewriter:
    def __init__(self, max_expansions: int = 3):
        self.max_expansions = max_expansions

    def clean_query(self, query: str) -> str:
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"^[.\-,;:!?]+", "", cleaned)
        cleaned = re.sub(r"[.\-,;:!?]+$", "", cleaned)
        return cleaned.strip()

    def expand_synonyms(self, query: str) -> list[str]:
        expansions = [query]
        tokens = re.findall(r"\w+|[^\w\s]", query.lower())
        for token in tokens:
            if token in TECH_SYNONYMS:
                for synonym in TECH_SYNONYMS[token]:
                    if synonym.lower() not in query.lower():
                        expanded = query + " " + synonym
                        if expanded not in expansions:
                            expansions.append(expanded)
                        if len(expansions) >= self.max_expansions + 1:
                            return expansions
        return expansions

    def remove_stop_words(self, query: str) -> str:
        tokens = query.split()
        filtered = []
        for token in tokens:
            lower = token.lower().strip(".,;:!?")
            if lower not in STOP_WORDS_EN and lower not in STOP_WORDS_ZH:
                filtered.append(token)
        result = " ".join(filtered)
        return result if result else query

    def expand_query(self, query: str) -> list[str]:
        try:
            cleaned = self.clean_query(query)
            if not cleaned:
                return [query]
            expansions = self.expand_synonyms(cleaned)
            core = self.remove_stop_words(cleaned)
            if core and core != cleaned and core not in expansions:
                expansions.append(core)
            return expansions[: self.max_expansions + 1]
        except Exception as e:
            raise RetrievalError(f"查询重写失败: {e}") from e
