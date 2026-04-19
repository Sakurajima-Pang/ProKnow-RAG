import re
from pathlib import Path

ALLOWED_EXTENSIONS: set[str] = {".md", ".pdf", ".py", ".js", ".ts", ".tex", ".txt", ".rst", ".html", ".json", ".yaml", ".yml", ".toml"}
MAX_FILE_SIZE_MB: int = 50


def validate_file_path(file_path: str, base_dir: str = "./data") -> Path:
    path = Path(file_path).resolve()
    base = Path(base_dir).resolve()
    if not str(path).startswith(str(base)):
        raise ValueError(f"路径越界访问: {file_path}")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型: {path.suffix}")
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"文件大小超过限制 ({size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB)")
    return path


def validate_query(query: str, max_length: int = 2000) -> str:
    if not query or not query.strip():
        raise ValueError("查询不能为空")
    if len(query) > max_length:
        raise ValueError(f"查询长度超过限制 ({len(query)} > {max_length})")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", query)
    return cleaned.strip()
