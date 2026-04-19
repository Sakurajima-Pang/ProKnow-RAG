import ast
from pathlib import Path

from proknow_rag.data_preparation.base import BaseParser, Document
from proknow_rag.common.exceptions import ParsingError


class CodeParser(BaseParser):
    def parse(self, file_path: str) -> list[Document]:
        path = Path(file_path)
        if not path.exists():
            raise ParsingError(f"文件不存在: {file_path}")
        try:
            source = path.read_text(encoding="utf-8")
        except Exception as e:
            raise ParsingError(f"读取文件失败: {file_path}, {e}")

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            raise ParsingError(f"解析Python代码失败: {file_path}, {e}")

        return self._extract_documents(tree, source, str(path))

    def _extract_documents(self, tree: ast.Module, source: str, file_path: str) -> list[Document]:
        documents = []
        source_lines = source.splitlines()
        imports = self._extract_imports(tree)

        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            documents.append(Document(
                content=module_docstring,
                metadata={
                    "type": "module_docstring",
                    "function_name": "",
                    "class_name": "",
                    "line_number": 1,
                    "imports": imports,
                },
                source=file_path,
            ))

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = self._function_to_document(node, source_lines, imports, file_path)
                documents.append(doc)
            elif isinstance(node, ast.ClassDef):
                class_docs = self._class_to_documents(node, source_lines, imports, file_path)
                documents.extend(class_docs)

        return documents

    def _function_to_document(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_lines: list[str],
        imports: list[str],
        file_path: str,
    ) -> Document:
        signature = self._get_function_signature(node, source_lines)
        docstring = ast.get_docstring(node) or ""
        body_lines = source_lines[node.lineno - 1: node.end_lineno]
        content = "\n".join(body_lines)

        metadata = {
            "type": "function",
            "function_name": node.name,
            "class_name": "",
            "line_number": node.lineno,
            "end_line_number": node.end_lineno,
            "imports": imports,
            "signature": signature,
            "docstring": docstring,
        }
        return Document(content=content, metadata=metadata, source=file_path)

    def _class_to_documents(
        self,
        node: ast.ClassDef,
        source_lines: list[str],
        imports: list[str],
        file_path: str,
    ) -> list[Document]:
        documents = []
        class_docstring = ast.get_docstring(node) or ""
        class_header_lines = source_lines[node.lineno - 1: node.body[0].lineno - 1 if node.body else node.end_lineno]
        class_header = "\n".join(class_header_lines)

        if class_docstring:
            class_content = class_header + "\n" + class_docstring if class_header.strip() else class_docstring
        else:
            class_content = class_header

        documents.append(Document(
            content=class_content,
            metadata={
                "type": "class",
                "function_name": "",
                "class_name": node.name,
                "line_number": node.lineno,
                "end_line_number": node.end_lineno,
                "imports": imports,
                "docstring": class_docstring,
            },
            source=file_path,
        ))

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = self._method_to_document(item, node.name, source_lines, imports, file_path)
                documents.append(doc)

        return documents

    def _method_to_document(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        class_name: str,
        source_lines: list[str],
        imports: list[str],
        file_path: str,
    ) -> Document:
        signature = self._get_function_signature(node, source_lines)
        docstring = ast.get_docstring(node) or ""
        body_lines = source_lines[node.lineno - 1: node.end_lineno]
        content = "\n".join(body_lines)

        metadata = {
            "type": "method",
            "function_name": node.name,
            "class_name": class_name,
            "line_number": node.lineno,
            "end_line_number": node.end_lineno,
            "imports": imports,
            "signature": signature,
            "docstring": docstring,
        }
        return Document(content=content, metadata=metadata, source=file_path)

    def _get_function_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef, source_lines: list[str]) -> str:
        sig_end = node.body[0].lineno - 1 if node.body else node.end_lineno
        sig_lines = source_lines[node.lineno - 1: sig_end]
        sig_text = "\n".join(sig_lines)
        first_line = sig_lines[0] if sig_lines else ""
        if ":" in first_line:
            return first_line[: first_line.index(":")].strip()
        return sig_text.strip()

    def _extract_imports(self, tree: ast.Module) -> list[str]:
        imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
        return imports
