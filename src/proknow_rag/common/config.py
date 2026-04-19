from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_storage_path: str = "./qdrant_storage"
    bge_m3_model_path: str = "./models/bge-m3"
    hf_endpoint: str = "https://hf-mirror.com"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
