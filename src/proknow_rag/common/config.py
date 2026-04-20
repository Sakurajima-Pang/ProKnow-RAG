from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    qdrant_storage_path: str = str(_PROJECT_ROOT / "qdrant_storage")
    bge_m3_model_path: str = str(_PROJECT_ROOT / "models" / "bge-m3")
    bge_reranker_model_path: str = str(_PROJECT_ROOT / "models" / "bge-reranker-v2-m3")
    data_dir: str = str(_PROJECT_ROOT / "data" / "raw")
    hf_endpoint: str = "https://hf-mirror.com"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @model_validator(mode="after")
    def resolve_paths(self) -> "Settings":
        for field in ("qdrant_storage_path", "bge_m3_model_path", "bge_reranker_model_path", "data_dir"):
            value = getattr(self, field)
            p = Path(value)
            if not p.is_absolute():
                setattr(self, field, str((_PROJECT_ROOT / p).resolve()))
        return self
