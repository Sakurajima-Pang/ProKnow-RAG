import re
from pathlib import Path

from proknow_rag.data_preparation.base import BaseParser, Document
from proknow_rag.common.exceptions import ParsingError
from proknow_rag.data_preparation.parsers.pdf_parser import PdfParser


_SECTION_RE = re.compile(r"\\(section|subsection|subsubsection|paragraph|subparagraph)\{([^}]+)\}")
_ABSTRACT_RE = re.compile(r"\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}")
_TITLE_RE = re.compile(r"\\title\{([^}]+)\}")
_BEGIN_DOC_RE = re.compile(r"\\begin\{document\}")
_END_DOC_RE = re.compile(r"\\end\{document\}")

_SECTION_MAPPING = {
    "section": 1,
    "subsection": 2,
    "subsubsection": 3,
    "paragraph": 4,
    "subparagraph": 5,
}

_KEYWORD_MAPPING = {
    "abstract": "摘要",
    "introduction": "引言",
    "intro": "引言",
    "method": "方法",
    "methodology": "方法",
    "approach": "方法",
    "experiment": "实验",
    "experiments": "实验",
    "evaluation": "实验",
    "result": "结果",
    "results": "结果",
    "conclusion": "结论",
    "conclusions": "结论",
    "discussion": "讨论",
    "related work": "相关工作",
    "relatedwork": "相关工作",
    "background": "背景",
    "reference": "参考文献",
    "references": "参考文献",
    "bibliography": "参考文献",
    "acknowledgment": "致谢",
    "acknowledgements": "致谢",
}


class PaperParser(BaseParser):
    def __init__(self, extract_tables: bool = True):
        self.pdf_parser = PdfParser(extract_tables=extract_tables)
        self.extract_tables = extract_tables

    def parse(self, file_path: str) -> list[Document]:
        path = Path(file_path)
        if not path.exists():
            raise ParsingError(f"文件不存在: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".tex":
            return self._parse_latex(str(path))
        elif suffix == ".pdf":
            return self.pdf_parser.parse(str(path))
        else:
            raise ParsingError(f"不支持的论文格式: {suffix}")

    def _parse_latex(self, file_path: str) -> list[Document]:
        try:
            text = Path(file_path).read_text(encoding="utf-8")
        except Exception as e:
            raise ParsingError(f"读取LaTeX文件失败: {file_path}, {e}")

        documents = []

        title_match = _TITLE_RE.search(text)
        if title_match:
            documents.append(Document(
                content=title_match.group(1).strip(),
                metadata={
                    "section_name": "title",
                    "section_level": 0,
                    "section_path": "title",
                    "section_type": "标题",
                },
                source=file_path,
            ))

        abstract_match = _ABSTRACT_RE.search(text)
        if abstract_match:
            abstract_text = abstract_match.group(1).strip()
            documents.append(Document(
                content=abstract_text,
                metadata={
                    "section_name": "abstract",
                    "section_level": 1,
                    "section_path": "abstract",
                    "section_type": "摘要",
                },
                source=file_path,
            ))

        doc_match = _BEGIN_DOC_RE.search(text)
        if doc_match:
            body_text = text[doc_match.end():]
            end_match = _END_DOC_RE.search(body_text)
            if end_match:
                body_text = body_text[:end_match.start()]
        else:
            body_text = text

        sections = self._split_latex_sections(body_text, file_path)
        documents.extend(sections)

        return documents

    def _split_latex_sections(self, text: str, file_path: str) -> list[Document]:
        sections = []
        matches = list(_SECTION_RE.finditer(text))

        if not matches:
            cleaned = self._clean_latex(text)
            if cleaned.strip():
                sections.append(Document(
                    content=cleaned.strip(),
                    metadata={
                        "section_name": "body",
                        "section_level": 0,
                        "section_path": "body",
                        "section_type": "",
                    },
                    source=file_path,
                ))
            return sections

        path_stack: list[tuple[int, str]] = []
        for i, match in enumerate(matches):
            cmd = match.group(1)
            title = match.group(2).strip()
            level = _SECTION_MAPPING.get(cmd, 1)

            while path_stack and path_stack[-1][0] >= level:
                path_stack.pop()
            path_stack.append((level, title))
            section_path = " > ".join(t for _, t in path_stack)

            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end]
            cleaned = self._clean_latex(content)

            section_type = self._classify_section(title)

            sections.append(Document(
                content=cleaned.strip(),
                metadata={
                    "section_name": title,
                    "section_level": level,
                    "section_path": section_path,
                    "section_type": section_type,
                },
                source=file_path,
            ))

        return sections

    def _clean_latex(self, text: str) -> str:
        text = re.sub(r"\\label\{[^}]*\}", "", text)
        text = re.sub(r"\\ref\{[^}]*\}", "", text)
        text = re.sub(r"\\cite\{[^}]*\}", "", text)
        text = re.sub(r"\\begin\{[^}]*\}", "", text)
        text = re.sub(r"\\end\{[^}]*\}", "", text)
        text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\[a-zA-Z]+", "", text)
        text = re.sub(r"[{}]", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _classify_section(self, title: str) -> str:
        lower = title.lower().strip()
        return _KEYWORD_MAPPING.get(lower, "")
