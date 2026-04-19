import re

from proknow_rag.common.exceptions import QueryValidationError
from proknow_rag.data_preparation.validators import validate_query as _base_validate_query

PROMPT_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+(instructions?|prompts?)", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?previous\s+(instructions?|context)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"pretend\s+you\s+are", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+you\s+are", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"sudo\s+mode", re.IGNORECASE),
    re.compile(r"developer\s+mode", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
    re.compile(r"override\s+(safety|security|filter)", re.IGNORECASE),
    re.compile(r"bypass\s+(safety|security|filter|restriction)", re.IGNORECASE),
    re.compile(r"reveal\s+(your|the)\s+(system|initial|original)\s+(prompt|instruction)", re.IGNORECASE),
    re.compile(r"output\s+your\s+(system|initial)\s+prompt", re.IGNORECASE),
]


def validate_query(query: str, max_length: int = 2000) -> str:
    try:
        cleaned = _base_validate_query(query, max_length=max_length)
    except ValueError as e:
        raise QueryValidationError(str(e)) from e
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def detect_prompt_injection(query: str) -> tuple[bool, list[str]]:
    detected_patterns: list[str] = []
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(query):
            detected_patterns.append(pattern.pattern)
    return len(detected_patterns) > 0, detected_patterns


def preprocess_query(query: str) -> str:
    processed = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)
    processed = re.sub(r"\s+", " ", processed).strip()
    processed = re.sub(r"^[.\-,;:!?]+", "", processed)
    processed = re.sub(r"[.\-,;:!?]+$", "", processed)
    return processed.strip()


def validate_and_sanitize(query: str, max_length: int = 2000) -> str:
    is_injection, patterns = detect_prompt_injection(query)
    if is_injection:
        raise QueryValidationError(f"检测到潜在的 Prompt 注入: 匹配模式 {patterns}")
    return validate_query(query, max_length=max_length)
