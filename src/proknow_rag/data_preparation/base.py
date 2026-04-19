from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class Document(BaseModel):
    content: str
    metadata: dict = {}
    source: str


class PreparedChunk(BaseModel):
    content: str
    metadata: dict = {}
    source: str
    chunk_type: str = "text"
    doc_hash: str = ""


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> list[Document]:
        ...


class BaseChunker(ABC):
    def __init__(self, overlap: int = 50):
        self.overlap = overlap

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[PreparedChunk]:
        ...
