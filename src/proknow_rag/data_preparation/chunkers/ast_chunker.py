import ast
import hashlib
from pathlib import Path

from proknow_rag.data_preparation.base import BaseChunker, Document, PreparedChunk
from proknow_rag.common.exceptions import ChunkingError


class AstChunker(BaseChunker):
    def __init__(
        self,
        max_class_size: int = 3000,
        overlap: int = 50,
    ):
        super().__init__(overlap=overlap)
        self.max_class_size = max_class_size

    def chunk(self, documents: list[Document]) -> list[PreparedChunk]:
        chunks = []
        for doc in documents:
            try:
                doc_chunks = self._chunk_document(doc)
                chunks.extend(doc_chunks)
            except Exception as e:
                raise ChunkingError(f"AST分块失败: {doc.source}, {e}")
        return chunks

    def _chunk_document(self, doc: Document) -> list[PreparedChunk]:
        doc_type = doc.metadata.get("type", "")

        if doc_type in ("function", "method"):
            return [self._function_to_chunk(doc)]
        elif doc_type == "class":
            return self._class_to_chunks(doc)
        elif doc_type == "module_docstring":
            return [self._docstring_to_chunk(doc)]
        else:
            return self._generic_chunk(doc)

    def _function_to_chunk(self, doc: Document) -> PreparedChunk:
        content = doc.content
        doc_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        return PreparedChunk(
            content=content,
            metadata={
                **doc.metadata,
                "chunk_type_name": "function",
                "chunk_index": 0,
                "total_chunks": 1,
            },
            source=doc.source,
            chunk_type="ast_function",
            doc_hash=doc_hash,
        )

    def _class_to_chunks(self, doc: Document) -> list[PreparedChunk]:
        class_name = doc.metadata.get("class_name", "")
        class_content = doc.content

        if len(class_content) <= self.max_class_size:
            doc_hash = hashlib.md5(class_content.encode("utf-8")).hexdigest()
            return [PreparedChunk(
                content=class_content,
                metadata={
                    **doc.metadata,
                    "chunk_type_name": "class",
                    "chunk_index": 0,
                    "total_chunks": 1,
                },
                source=doc.source,
                chunk_type="ast_class",
                doc_hash=doc_hash,
            )]

        return self._split_class_by_methods(doc)

    def _split_class_by_methods(self, doc: Document) -> list[PreparedChunk]:
        chunks = []
        class_name = doc.metadata.get("class_name", "")
        source = doc.content
        lines = source.splitlines()

        try:
            tree = ast.parse(source)
        except SyntaxError:
            doc_hash = hashlib.md5(source.encode("utf-8")).hexdigest()
            return [PreparedChunk(
                content=source,
                metadata={**doc.metadata, "chunk_index": 0, "total_chunks": 1},
                source=doc.source,
                chunk_type="ast_class",
                doc_hash=doc_hash,
            )]

        class_header_end = 0
        class_node = None
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                class_node = node
                if node.body:
                    first_body = node.body[0]
                    class_header_end = first_body.lineno - 1
                break

        if class_node is None:
            doc_hash = hashlib.md5(source.encode("utf-8")).hexdigest()
            return [PreparedChunk(
                content=source,
                metadata={**doc.metadata, "chunk_index": 0, "total_chunks": 1},
                source=doc.source,
                chunk_type="ast_class",
                doc_hash=doc_hash,
            )]

        if class_header_end > 0:
            header_lines = lines[:class_header_end]
            header_text = "\n".join(header_lines)
            doc_hash = hashlib.md5(header_text.encode("utf-8")).hexdigest()
            chunks.append(PreparedChunk(
                content=header_text,
                metadata={
                    **doc.metadata,
                    "chunk_type_name": "class_header",
                    "chunk_index": len(chunks),
                },
                source=doc.source,
                chunk_type="ast_class_header",
                doc_hash=doc_hash,
            ))

        for item in class_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_lines = lines[item.lineno - 1: item.end_lineno]
                method_text = "\n".join(method_lines)
                doc_hash = hashlib.md5(method_text.encode("utf-8")).hexdigest()

                method_signature = self._get_signature(item, lines)
                method_docstring = ast.get_docstring(item) or ""

                chunks.append(PreparedChunk(
                    content=method_text,
                    metadata={
                        **doc.metadata,
                        "type": "method",
                        "function_name": item.name,
                        "class_name": class_name,
                        "line_number": item.lineno,
                        "end_line_number": item.end_lineno,
                        "signature": method_signature,
                        "docstring": method_docstring,
                        "chunk_type_name": "method",
                        "chunk_index": len(chunks),
                    },
                    source=doc.source,
                    chunk_type="ast_method",
                    doc_hash=doc_hash,
                ))
            else:
                other_lines = lines[item.lineno - 1: item.end_lineno] if hasattr(item, "end_lineno") else []
                if other_lines:
                    other_text = "\n".join(other_lines)
                    doc_hash = hashlib.md5(other_text.encode("utf-8")).hexdigest()
                    chunks.append(PreparedChunk(
                        content=other_text,
                        metadata={
                            **doc.metadata,
                            "chunk_type_name": "class_body",
                            "chunk_index": len(chunks),
                        },
                        source=doc.source,
                        chunk_type="ast_class_body",
                        doc_hash=doc_hash,
                    ))

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def _docstring_to_chunk(self, doc: Document) -> PreparedChunk:
        doc_hash = hashlib.md5(doc.content.encode("utf-8")).hexdigest()
        return PreparedChunk(
            content=doc.content,
            metadata={
                **doc.metadata,
                "chunk_index": 0,
                "total_chunks": 1,
            },
            source=doc.source,
            chunk_type="ast_docstring",
            doc_hash=doc_hash,
        )

    def _generic_chunk(self, doc: Document) -> list[PreparedChunk]:
        doc_hash = hashlib.md5(doc.content.encode("utf-8")).hexdigest()
        return [PreparedChunk(
            content=doc.content,
            metadata={**doc.metadata, "chunk_index": 0, "total_chunks": 1},
            source=doc.source,
            chunk_type="ast_generic",
            doc_hash=doc_hash,
        )]

    def _get_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef, source_lines: list[str]) -> str:
        sig_end = node.body[0].lineno - 1 if node.body else node.end_lineno
        sig_lines = source_lines[node.lineno - 1: sig_end]
        first_line = sig_lines[0] if sig_lines else ""
        if ":" in first_line:
            return first_line[: first_line.index(":")].strip()
        return "\n".join(sig_lines).strip()
