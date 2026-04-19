import json
from pathlib import Path

from proknow_rag.data_preparation.base import BaseParser, Document, PreparedChunk
from proknow_rag.common.exceptions import ParsingError


class LlmParser(BaseParser):
    def parse(self, file_path: str) -> list[Document]:
        path = Path(file_path)
        if not path.exists():
            raise ParsingError(f"文件不存在: {file_path}")

        data = self._load_json(path)
        chunks = self._validate_and_extract(data, str(path))
        return chunks

    def parse_batch(self, file_paths: list[str]) -> list[Document]:
        documents = []
        for fp in file_paths:
            documents.extend(self.parse(fp))
        return documents

    def _load_json(self, path: Path) -> list[dict]:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            raise ParsingError(f"读取JSON文件失败: {path}, {e}")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ParsingError(f"JSON格式错误: {path}, {e}")

        if isinstance(data, dict):
            if "chunks" in data:
                data = data["chunks"]
            elif "documents" in data:
                data = data["documents"]
            else:
                data = [data]

        if not isinstance(data, list):
            raise ParsingError(f"JSON内容应为列表或包含chunks/documents字段的对象: {path}")

        return data

    def _validate_and_extract(self, data: list[dict], source: str) -> list[Document]:
        documents = []
        for i, item in enumerate(data):
            try:
                chunk = PreparedChunk(**item)
                documents.append(Document(
                    content=chunk.content,
                    metadata={**chunk.metadata, "chunk_type": chunk.chunk_type, "doc_hash": chunk.doc_hash},
                    source=chunk.source or source,
                ))
            except Exception as e:
                raise ParsingError(f"第{i}个chunk验证失败: {e}")
        return documents
