import re
from pathlib import Path

import markdown
from markdown.treeprocessors import Treeprocessor
from markdown.extensions import Extension

from proknow_rag.data_preparation.base import BaseParser, Document
from proknow_rag.common.exceptions import ParsingError


class _StructureExtractor(Treeprocessor):
    def __init__(self, md_parser):
        super().__init__()
        self.md_parser = md_parser

    def run(self, root):
        self.md_parser._tree = root
        return None


class _StructureExtension(Extension):
    def __init__(self, md_parser):
        self.md_parser = md_parser
        super().__init__()

    def extendMarkdown(self, md):
        md.treeprocessors.register(
            _StructureExtractor(self.md_parser), "structure_extractor", 0
        )


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_TABLE_RE = re.compile(r"(\|.+\|[\r\n]+\|[-:\s|]+\|[\r\n]+((?:\|.+\|[\r\n]*)+))", re.MULTILINE)
_LIST_RE = re.compile(r"^(\s*[-*+]|\s*\d+\.)\s+.+$", re.MULTILINE)


class MarkdownParser(BaseParser):
    def __init__(self):
        self._tree = None

    def parse(self, file_path: str) -> list[Document]:
        path = Path(file_path)
        if not path.exists():
            raise ParsingError(f"文件不存在: {file_path}")
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            raise ParsingError(f"读取文件失败: {file_path}, {e}")

        return self._extract_documents(text, str(path))

    def _extract_documents(self, text: str, source: str) -> list[Document]:
        documents = []
        sections = self._split_by_headings(text)
        for section in sections:
            content = section["content"].strip()
            if not content:
                continue
            is_code = self._is_code_block(content)
            metadata = {
                "heading_level": section["level"],
                "section_path": section["path"],
                "is_code_block": is_code,
            }
            documents.append(Document(content=content, metadata=metadata, source=source))
        return documents

    def _split_by_headings(self, text: str) -> list[dict]:
        headings = list(_HEADING_RE.finditer(text))
        if not headings:
            return [{"content": text, "level": 0, "path": ""}]

        sections = []
        if headings[0].start() > 0:
            preamble = text[: headings[0].start()].strip()
            if preamble:
                sections.append({"content": preamble, "level": 0, "path": ""})

        path_stack: list[tuple[int, str]] = []
        for i, match in enumerate(headings):
            level = len(match.group(1))
            title = match.group(2).strip()

            while path_stack and path_stack[-1][0] >= level:
                path_stack.pop()
            path_stack.append((level, title))
            section_path = " > ".join(t for _, t in path_stack)

            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            content = text[start:end]

            sections.append({
                "content": content,
                "level": level,
                "path": section_path,
            })

        return sections

    def _is_code_block(self, content: str) -> bool:
        stripped = content.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            return True
        code_matches = _CODE_BLOCK_RE.findall(content)
        total_code_len = sum(len(m) for m in code_matches)
        return total_code_len > len(content) * 0.8
