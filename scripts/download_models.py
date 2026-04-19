import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_INFO = {
    "BAAI/bge-m3": {
        "local_dir": "./models/bge-m3",
        "allow_patterns": [
            "*.json", "*.txt", "*.model", "*.safetensors", "*.bin",
            "tokenizer*", "special_tokens_map*", "config*",
        ],
    },
    "BAAI/bge-reranker-v2-m3": {
        "local_dir": "./models/bge-reranker-v2-m3",
        "allow_patterns": [
            "*.json", "*.txt", "*.model", "*.safetensors", "*.bin",
            "tokenizer*", "special_tokens_map*", "config*",
        ],
    },
}


def download_model(model_name: str, local_dir: str, allow_patterns: list[str] | None = None):
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_ENDPOINT"] = hf_endpoint

    print(f"下载模型 {model_name} -> {local_dir}")
    print(f"使用 HuggingFace 镜像: {hf_endpoint}")

    Path(local_dir).mkdir(parents=True, exist_ok=True)

    kwargs = {
        "repo_id": model_name,
        "local_dir": local_dir,
    }
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns

    snapshot_download(**kwargs)
    print(f"模型 {model_name} 下载完成")


def main():
    for model_name, info in MODEL_INFO.items():
        download_model(
            model_name,
            info["local_dir"],
            info.get("allow_patterns"),
        )
    print("所有模型下载完成")


if __name__ == "__main__":
    main()
