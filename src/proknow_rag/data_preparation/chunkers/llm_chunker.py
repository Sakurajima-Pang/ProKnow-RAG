import json
import hashlib
from pathlib import Path

from proknow_rag.data_preparation.base import BaseChunker, Document, PreparedChunk
from proknow_rag.common.exceptions import ChunkingError


class LlmChunker(BaseChunker):
    def chunk(self, documents: list[Document]) -> list[PreparedChunk]:
        chunks = []
        for doc in documents:
            try:
                doc_chunks = self._chunk_document(doc)
                chunks.extend(doc_chunks)
            except Exception as e:
                raise ChunkingError(f"LLM分块失败: {doc.source}, {e}")
        return chunks

    def _chunk_document(self, doc: Document) -> list[PreparedChunk]:
        path = Path(doc.source)
        if not path.exists():
            raise ChunkingError(f"外部JSON文件不存在: {doc.source}")

        data = self._load_json(path)
        return self._validate_and_extract(data)

    def _load_json(self, path: Path) -> list[dict]:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            raise ChunkingError(f"读取JSON文件失败: {path}, {e}")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ChunkingError(f"JSON格式错误: {path}, {e}")

        if isinstance(data, dict):
            if "chunks" in data:
                data = data["chunks"]
            elif "documents" in data:
                data = data["documents"]
            else:
                data = [data]

        if not isinstance(data, list):
            raise ChunkingError(f"JSON内容应为列表或包含chunks/documents字段的对象: {path}")

        return data

    def _validate_and_extract(self, data: list[dict]) -> list[PreparedChunk]:
        chunks = []
        for i, item in enumerate(data):
            try:
                chunk = PreparedChunk(**item)
                if not chunk.doc_hash:
                    chunk.doc_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
                chunks.append(chunk)
            except Exception as e:
                raise ChunkingError(f"第{i}个chunk验证失败: {e}")
        return chunks

    def load_from_path(self, file_path: str) -> list[PreparedChunk]:
        path = Path(file_path)
        if not path.exists():
            raise ChunkingError(f"文件不存在: {file_path}")

        data = self._load_json(path)
        return self._validate_and_extract(data)
