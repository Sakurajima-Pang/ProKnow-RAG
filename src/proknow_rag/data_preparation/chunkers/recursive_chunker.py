import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from proknow_rag.data_preparation.base import BaseChunker, Document, PreparedChunk
from proknow_rag.common.exceptions import ChunkingError


class RecursiveChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
        separators: list[str] | None = None,
    ):
        super().__init__(overlap=overlap)
        self.chunk_size = chunk_size
        self.separators = separators or ["\n\n", "\n", "。", ".", " ", ""]

    def chunk(self, documents: list[Document]) -> list[PreparedChunk]:
        chunks = []
        for doc in documents:
            try:
                doc_chunks = self._chunk_document(doc)
                chunks.extend(doc_chunks)
            except Exception as e:
                raise ChunkingError(f"分块失败: {doc.source}, {e}")
        return chunks

    def _chunk_document(self, doc: Document) -> list[PreparedChunk]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=self.separators,
        )

        texts = splitter.split_text(doc.content)
        chunks = []
        for i, text in enumerate(texts):
            doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            metadata = {**doc.metadata, "chunk_index": i, "total_chunks": len(texts)}
            chunks.append(PreparedChunk(
                content=text,
                metadata=metadata,
                source=doc.source,
                chunk_type="recursive",
                doc_hash=doc_hash,
            ))
        return chunks
