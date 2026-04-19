from pathlib import Path

from pypdf import PdfReader

from proknow_rag.data_preparation.base import BaseParser, Document
from proknow_rag.common.exceptions import ParsingError


class PdfParser(BaseParser):
    def __init__(self, extract_tables: bool = True):
        self.extract_tables = extract_tables

    def parse(self, file_path: str) -> list[Document]:
        path = Path(file_path)
        if not path.exists():
            raise ParsingError(f"文件不存在: {file_path}")
        try:
            reader = PdfReader(str(path))
        except Exception as e:
            raise ParsingError(f"读取PDF失败: {file_path}, {e}")

        documents = []
        total_pages = len(reader.pages)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            metadata = {
                "page_number": page_num,
                "total_pages": total_pages,
                "source_file": str(path),
            }

            if self.extract_tables:
                tables = self._extract_tables_from_page(page)
                if tables:
                    metadata["has_table"] = True
                    metadata["table_count"] = len(tables)

            documents.append(Document(content=text.strip(), metadata=metadata, source=str(path)))

        return documents

    def _extract_tables_from_page(self, page) -> list[str]:
        tables = []
        if not hasattr(page, "tables"):
            return tables
        try:
            for table in page.tables:
                rows = []
                for row in table.rows:
                    cells = [cell.extract_text().strip() for cell in row.cells]
                    rows.append(" | ".join(cells))
                if rows:
                    tables.append("\n".join(rows))
        except Exception:
            pass
        return tables
