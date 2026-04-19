class ProKnowRAGError(Exception):
    pass


class DataPreparationError(ProKnowRAGError):
    pass


class ParsingError(DataPreparationError):
    pass


class ChunkingError(DataPreparationError):
    pass


class IndexConstructionError(ProKnowRAGError):
    pass


class EmbeddingError(IndexConstructionError):
    pass


class QdrantError(IndexConstructionError):
    pass


class RetrievalError(ProKnowRAGError):
    pass


class RerankerError(RetrievalError):
    pass


class QueryValidationError(RetrievalError):
    pass
