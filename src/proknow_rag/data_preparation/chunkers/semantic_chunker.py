import hashlib
import re

import numpy as np

from proknow_rag.data_preparation.base import BaseChunker, Document, PreparedChunk
from proknow_rag.common.exceptions import ChunkingError


class SemanticChunker(BaseChunker):
    def __init__(
        self,
        embed_fn=None,
        similarity_threshold: float = 0.5,
        overlap: int = 50,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ):
        super().__init__(overlap=overlap)
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, documents: list[Document]) -> list[PreparedChunk]:
        chunks = []
        for doc in documents:
            try:
                doc_chunks = self._chunk_document(doc)
                chunks.extend(doc_chunks)
            except Exception as e:
                raise ChunkingError(f"语义分块失败: {doc.source}, {e}")
        return chunks

    def _chunk_document(self, doc: Document) -> list[PreparedChunk]:
        sentences = self._split_sentences(doc.content)
        if len(sentences) <= 1:
            doc_hash = hashlib.sha256(doc.content.encode("utf-8")).hexdigest()
            return [PreparedChunk(
                content=doc.content,
                metadata={**doc.metadata, "chunk_index": 0, "total_chunks": 1},
                source=doc.source,
                chunk_type="semantic",
                doc_hash=doc_hash,
            )]

        if self.embed_fn is None:
            return self._chunk_by_length(sentences, doc)

        embeddings = self._get_embeddings(sentences)
        split_points = self._find_split_points(embeddings)

        return self._build_chunks(sentences, split_points, doc)

    def _split_sentences(self, text: str) -> list[str]:
        pattern = r'(?<=[。！？.!?])\s*|(?<=\n)\s*'
        parts = re.split(pattern, text)
        return [s.strip() for s in parts if s.strip()]

    def _get_embeddings(self, sentences: list[str]) -> np.ndarray:
        try:
            embeddings = self.embed_fn(sentences)
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            return embeddings
        except Exception as e:
            raise ChunkingError(f"生成嵌入失败: {e}")

    def _find_split_points(self, embeddings: np.ndarray) -> list[int]:
        split_points = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self.similarity_threshold:
                split_points.append(i + 1)
        return split_points

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _build_chunks(
        self,
        sentences: list[str],
        split_points: list[int],
        doc: Document,
    ) -> list[PreparedChunk]:
        chunks = []
        start = 0
        boundaries = split_points + [len(sentences)]

        for boundary in boundaries:
            end = boundary
            chunk_text = " ".join(sentences[start:end])

            if len(chunk_text) < self.min_chunk_size and boundary < len(sentences):
                continue

            if len(chunk_text) > self.max_chunk_size:
                sub_chunks = self._split_by_size(chunk_text, doc, len(chunks))
                chunks.extend(sub_chunks)
            else:
                doc_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                chunks.append(PreparedChunk(
                    content=chunk_text,
                    metadata={**doc.metadata, "chunk_index": len(chunks), "sentence_range": (start, end)},
                    source=doc.source,
                    chunk_type="semantic",
                    doc_hash=doc_hash,
                ))

            start = end

        if start < len(sentences):
            remaining = " ".join(sentences[start:])
            doc_hash = hashlib.sha256(remaining.encode("utf-8")).hexdigest()
            chunks.append(PreparedChunk(
                content=remaining,
                metadata={**doc.metadata, "chunk_index": len(chunks), "sentence_range": (start, len(sentences))},
                source=doc.source,
                chunk_type="semantic",
                doc_hash=doc_hash,
            ))

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def _chunk_by_length(self, sentences: list[str], doc: Document) -> list[PreparedChunk]:
        chunks = []
        current_text = ""

        for sentence in sentences:
            if len(current_text) + len(sentence) > self.max_chunk_size and current_text:
                doc_hash = hashlib.sha256(current_text.encode("utf-8")).hexdigest()
                chunks.append(PreparedChunk(
                    content=current_text.strip(),
                    metadata={**doc.metadata, "chunk_index": len(chunks)},
                    source=doc.source,
                    chunk_type="semantic_length",
                    doc_hash=doc_hash,
                ))
                overlap_text = current_text[-self.overlap:] if self.overlap > 0 else ""
                current_text = overlap_text + " " + sentence
            else:
                current_text = current_text + " " + sentence if current_text else sentence

        if current_text.strip():
            doc_hash = hashlib.sha256(current_text.encode("utf-8")).hexdigest()
            chunks.append(PreparedChunk(
                content=current_text.strip(),
                metadata={**doc.metadata, "chunk_index": len(chunks)},
                source=doc.source,
                chunk_type="semantic_length",
                doc_hash=doc_hash,
            ))

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def _split_by_size(self, text: str, doc: Document, start_index: int) -> list[PreparedChunk]:
        chunks = []
        while len(text) > self.max_chunk_size:
            split_pos = text.rfind("。", 0, self.max_chunk_size)
            if split_pos == -1:
                split_pos = text.rfind(" ", 0, self.max_chunk_size)
            if split_pos == -1:
                split_pos = self.max_chunk_size

            part = text[:split_pos + 1].strip()
            doc_hash = hashlib.sha256(part.encode("utf-8")).hexdigest()
            chunks.append(PreparedChunk(
                content=part,
                metadata={**doc.metadata, "chunk_index": start_index + len(chunks)},
                source=doc.source,
                chunk_type="semantic",
                doc_hash=doc_hash,
            ))
            text = text[split_pos + 1:]

        if text.strip():
            doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunks.append(PreparedChunk(
                content=text.strip(),
                metadata={**doc.metadata, "chunk_index": start_index + len(chunks)},
                source=doc.source,
                chunk_type="semantic",
                doc_hash=doc_hash,
            ))

        return chunks
