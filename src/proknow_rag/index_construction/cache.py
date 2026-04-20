import json
from pathlib import Path

import numpy as np

from proknow_rag.common.config import Settings


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


class EmbeddingCache:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._cache_dir = Path(self.settings.qdrant_storage_path) / "embedding_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / "embeddings.jsonl"
        self._cache: dict[str, dict] = {}
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
                        self._cache[content_hash] = entry.get("data", {})
                except (json.JSONDecodeError, KeyError):
                    continue

    def _append_entry(self, content_hash: str, data: dict) -> None:
        entry = {"hash": content_hash, "data": data}
        with open(self._cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, cls=_NumpyEncoder) + "\n")

    def get(self, content_hash: str) -> dict | None:
        return self._cache.get(content_hash)

    def put(self, content_hash: str, data: dict) -> None:
        self._cache[content_hash] = data
        self._append_entry(content_hash, data)

    def invalidate(self, content_hash: str) -> None:
        if content_hash in self._cache:
            del self._cache[content_hash]
            self._rebuild_file()

    def clear(self) -> None:
        self._cache.clear()
        if self._cache_file.exists():
            self._cache_file.write_text("", encoding="utf-8")

    def _rebuild_file(self) -> None:
        with open(self._cache_file, "w", encoding="utf-8") as f:
            for content_hash, data in self._cache.items():
                entry = {"hash": content_hash, "data": data}
                f.write(json.dumps(entry, ensure_ascii=False, cls=_NumpyEncoder) + "\n")

    def contains(self, content_hash: str) -> bool:
        return content_hash in self._cache

    def size(self) -> int:
        return len(self._cache)
