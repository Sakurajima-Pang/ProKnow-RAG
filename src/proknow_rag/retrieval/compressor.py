import re

from proknow_rag.common.exceptions import RetrievalError


class ContextCompressor:
    def __init__(self, min_relevance_score: float = 0.3, max_sentences: int = 20):
        self.min_relevance_score = min_relevance_score
        self.max_sentences = max_sentences

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[。！？.!?\n])", text)
        sentences = [s.strip() for s in parts if s.strip()]
        return sentences

    def _compute_relevance(self, sentence: str, query_terms: set[str]) -> float:
        if not query_terms or not sentence:
            return 0.0
        sentence_lower = sentence.lower()
        matched = sum(1 for term in query_terms if term.lower() in sentence_lower)
        total = len(query_terms)
        if total == 0:
            return 0.0
        return matched / total

    def _extract_key_sentences(self, sentences: list[str], query_terms: set[str]) -> list[str]:
        scored = []
        for i, sentence in enumerate(sentences):
            relevance = self._compute_relevance(sentence, query_terms)
            scored.append((i, sentence, relevance))
        scored.sort(key=lambda x: x[2], reverse=True)
        top = scored[: self.max_sentences]
        top.sort(key=lambda x: x[0])
        return [s for _, s, score in top if score >= self.min_relevance_score]

    def compress(self, document: str, query: str) -> str:
        try:
            if not document or not query:
                return document
            query_terms = set(re.findall(r"\w+", query.lower()))
            sentences = self._split_sentences(document)
            if not sentences:
                return document
            if len(sentences) <= 3:
                return document
            key_sentences = self._extract_key_sentences(sentences, query_terms)
            if not key_sentences:
                return " ".join(sentences[:3])
            return " ".join(key_sentences)
        except Exception as e:
            raise RetrievalError(f"上下文压缩失败: {e}") from e

    def compress_documents(self, documents: list[str], query: str) -> list[str]:
        return [self.compress(doc, query) for doc in documents]
