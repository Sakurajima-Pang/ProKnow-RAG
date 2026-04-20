import json
from pathlib import Path

from proknow_rag.common.config import Settings


class EmbeddingCache:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._cache_dir = Path(self.settings.qdrant_storage_path) / "embedding_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / "index_status.jsonl"
        self._cache: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self._cache_file.exists():
            return
        with open(self._cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    content_hash = entry.get("hash")
                    if content_hash:
                        self._cache.add(content_hash)
                except (json.JSONDecodeError, KeyError):
                    continue

    def get(self, content_hash: str) -> bool | None:
        if content_hash in self._cache:
            return True
        return None

    def put(self, content_hash: str, _data: object = None) -> None:
        if content_hash in self._cache:
            return
        self._cache.add(content_hash)
        with open(self._cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"hash": content_hash}, ensure_ascii=False) + "\n")

    def invalidate(self, content_hash: str) -> None:
        self._cache.discard(content_hash)
        self._rebuild_file()

    def clear(self) -> None:
        self._cache.clear()
        if self._cache_file.exists():
            self._cache_file.write_text("", encoding="utf-8")

    def _rebuild_file(self) -> None:
        with open(self._cache_file, "w", encoding="utf-8") as f:
            for content_hash in self._cache:
                f.write(json.dumps({"hash": content_hash}, ensure_ascii=False) + "\n")

    def contains(self, content_hash: str) -> bool:
        return content_hash in self._cache

    def size(self) -> int:
        return len(self._cache)
