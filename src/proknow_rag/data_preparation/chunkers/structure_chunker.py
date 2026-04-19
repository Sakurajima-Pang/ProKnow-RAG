import hashlib

from proknow_rag.data_preparation.base import BaseChunker, Document, PreparedChunk
from proknow_rag.common.exceptions import ChunkingError


class StructureChunker(BaseChunker):
    def __init__(
        self,
        max_chunk_size: int = 2000,
        overlap: int = 50,
        heading_metadata_key: str = "section_path",
        level_metadata_key: str = "heading_level",
    ):
        super().__init__(overlap=overlap)
        self.max_chunk_size = max_chunk_size
        self.heading_metadata_key = heading_metadata_key
        self.level_metadata_key = level_metadata_key

    def chunk(self, documents: list[Document]) -> list[PreparedChunk]:
        chunks = []
        for doc in documents:
            try:
                doc_chunks = self._chunk_document(doc)
                chunks.extend(doc_chunks)
            except Exception as e:
                raise ChunkingError(f"结构化分块失败: {doc.source}, {e}")
        return chunks

    def _chunk_document(self, doc: Document) -> list[PreparedChunk]:
        section_path = doc.metadata.get(self.heading_metadata_key, "")
        heading_level = doc.metadata.get(self.level_metadata_key, 0)

        if len(doc.content) <= self.max_chunk_size:
            doc_hash = hashlib.md5(doc.content.encode("utf-8")).hexdigest()
            return [PreparedChunk(
                content=doc.content,
                metadata={
                    **doc.metadata,
                    "section_path": section_path,
                    "heading_level": heading_level,
                    "chunk_index": 0,
                    "total_chunks": 1,
                },
                source=doc.source,
                chunk_type="structure",
                doc_hash=doc_hash,
            )]

        return self._split_large_section(doc, section_path, heading_level)

    def _split_large_section(self, doc: Document, section_path: str, heading_level: int) -> list[PreparedChunk]:
        paragraphs = self._split_paragraphs(doc.content)
        chunks = []
        current_text = ""
        chunk_index = 0

        for para in paragraphs:
            if len(current_text) + len(para) > self.max_chunk_size and current_text:
                doc_hash = hashlib.md5(current_text.encode("utf-8")).hexdigest()
                chunks.append(PreparedChunk(
                    content=current_text.strip(),
                    metadata={
                        **doc.metadata,
                        "section_path": section_path,
                        "heading_level": heading_level,
                        "chunk_index": chunk_index,
                    },
                    source=doc.source,
                    chunk_type="structure",
                    doc_hash=doc_hash,
                ))
                chunk_index += 1

                if self.overlap > 0:
                    overlap_text = current_text[-self.overlap:]
                    current_text = overlap_text + "\n\n" + para
                else:
                    current_text = para
            else:
                current_text = current_text + "\n\n" + para if current_text else para

        if current_text.strip():
            doc_hash = hashlib.md5(current_text.encode("utf-8")).hexdigest()
            chunks.append(PreparedChunk(
                content=current_text.strip(),
                metadata={
                    **doc.metadata,
                    "section_path": section_path,
                    "heading_level": heading_level,
                    "chunk_index": chunk_index,
                },
                source=doc.source,
                chunk_type="structure",
                doc_hash=doc_hash,
            ))

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]
