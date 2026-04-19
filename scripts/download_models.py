import hashlib
import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_INFO = {
    "BAAI/bge-m3": {
        "local_dir": "./models/bge-m3",
    },
    "BAAI/bge-reranker-v2-m3": {
        "local_dir": "./models/bge-reranker-v2-m3",
    },
}


def download_model(model_name: str, local_dir: str):
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_ENDPOINT"] = hf_endpoint

    print(f"下载模型 {model_name} -> {local_dir}")
    print(f"使用 HuggingFace 镜像: {hf_endpoint}")

    Path(local_dir).mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"模型 {model_name} 下载完成")


def main():
    for model_name, info in MODEL_INFO.items():
        download_model(model_name, info["local_dir"])
    print("所有模型下载完成")


if __name__ == "__main__":
    main()
