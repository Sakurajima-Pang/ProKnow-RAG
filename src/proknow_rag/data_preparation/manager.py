import hashlib
import json
from pathlib import Path

import structlog

from proknow_rag.data_preparation.base import Document, PreparedChunk
from proknow_rag.data_preparation.strategies import StrategyRegistry
from proknow_rag.data_preparation.validators import validate_file_path
from proknow_rag.common.exceptions import DataPreparationError, ParsingError, ChunkingError

logger = structlog.get_logger(__name__)


class DataManager:
    def __init__(
        self,
        strategy_registry: StrategyRegistry | None = None,
        config_path: str | None = None,
        processed_cache_path: str | None = None,
    ):
        self.registry = strategy_registry or StrategyRegistry()
        if config_path:
            self.registry.load_from_yaml(config_path)

        self._processed_hashes: set[str] = set()
        self._processed_cache_path = processed_cache_path
        if processed_cache_path:
            self._load_processed_cache(processed_cache_path)

    def process_file(self, file_path: str, base_dir: str = "./data") -> list[PreparedChunk]:
        path = validate_file_path(file_path, base_dir)
        extension = path.suffix.lower()

        if extension not in self.registry.supported_extensions:
            raise DataPreparationError(f"不支持的文件类型: {extension}")

        doc_hash = self._compute_file_hash(str(path))
        if doc_hash in self._processed_hashes:
            logger.info("跳过已处理文件", file_path=str(path), doc_hash=doc_hash)
            return []

        try:
            parser = self.registry.create_parser(extension)
            documents = parser.parse(str(path))
        except ParsingError:
            raise
        except Exception as e:
            raise ParsingError(f"解析文件失败: {path}, {e}")

        try:
            chunker = self.registry.create_chunker(extension)
            chunks = chunker.chunk(documents)
        except ChunkingError:
            raise
        except Exception as e:
            raise ChunkingError(f"分块失败: {path}, {e}")

        for chunk in chunks:
            if not chunk.doc_hash:
                chunk.doc_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()

        self._processed_hashes.add(doc_hash)
        if self._processed_cache_path:
            self._save_processed_cache(self._processed_cache_path)

        logger.info("文件处理完成", file_path=str(path), documents=len(documents), chunks=len(chunks))
        return chunks

    def process_directory(self, dir_path: str, recursive: bool = True) -> list[PreparedChunk]:
        path = Path(dir_path).resolve()
        if not path.exists() or not path.is_dir():
            raise DataPreparationError(f"目录不存在: {dir_path}")

        all_chunks = []
        files = self._collect_files(path, recursive)

        for file_path in files:
            try:
                chunks = self.process_file(str(file_path), str(path))
                all_chunks.extend(chunks)
            except (ParsingError, ChunkingError, DataPreparationError) as e:
                logger.warning("文件处理失败，跳过", file_path=str(file_path), error=str(e))
                continue

        logger.info("目录处理完成", dir_path=str(path), total_chunks=len(all_chunks))
        return all_chunks

    def process_batch(self, file_paths: list[str], base_dir: str = "./data") -> list[PreparedChunk]:
        all_chunks = []
        for file_path in file_paths:
            try:
                chunks = self.process_file(file_path, base_dir)
                all_chunks.extend(chunks)
            except (ParsingError, ChunkingError, DataPreparationError) as e:
                logger.warning("文件处理失败，跳过", file_path=file_path, error=str(e))
                continue
        return all_chunks

    def validate_chunks(self, chunks: list[PreparedChunk]) -> list[PreparedChunk]:
        valid_chunks = []
        for chunk in chunks:
            try:
                validated = PreparedChunk(**chunk.model_dump())
                valid_chunks.append(validated)
            except Exception as e:
                logger.warning("chunk验证失败", error=str(e), source=chunk.source)
        return valid_chunks

    def export_chunks(self, chunks: list[PreparedChunk], output_path: str, format: str = "json"):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            data = [chunk.model_dump() for chunk in chunks]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format == "jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(chunk.model_dump_json() + "\n")
        else:
            raise DataPreparationError(f"不支持的导出格式: {format}")

        logger.info("chunks已导出", output_path=str(path), count=len(chunks), format=format)

    def _collect_files(self, dir_path: Path, recursive: bool) -> list[Path]:
        files = []
        if recursive:
            for p in dir_path.rglob("*"):
                if p.is_file() and p.suffix.lower() in self.registry.supported_extensions:
                    files.append(p)
        else:
            for p in dir_path.iterdir():
                if p.is_file() and p.suffix.lower() in self.registry.supported_extensions:
                    files.append(p)
        return sorted(files)

    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_processed_cache(self, cache_path: str):
        path = Path(cache_path)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._processed_hashes = set(data)
                elif isinstance(data, dict) and "hashes" in data:
                    self._processed_hashes = set(data["hashes"])
            except Exception as e:
                logger.warning("缓存加载失败", error=str(e))
                self._processed_hashes = set()

    def _save_processed_cache(self, cache_path: str):
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"hashes": list(self._processed_hashes)}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @property
    def processed_count(self) -> int:
        return len(self._processed_hashes)

    def reset_cache(self):
        self._processed_hashes.clear()
        if self._processed_cache_path:
            self._save_processed_cache(self._processed_cache_path)
