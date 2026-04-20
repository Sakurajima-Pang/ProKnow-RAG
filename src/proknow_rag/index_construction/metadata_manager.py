import re

LICENSE_PATTERNS: dict[str, list[str]] = {
    "MIT": ["mit license", "permission is hereby granted, free of charge", "mit"],
    "Apache-2.0": ["apache license", "version 2.0", "licensed under the apache"],
    "GPL-3.0": ["gnu general public license", "gpl", "version 3"],
    "GPL-2.0": ["gnu general public license", "gpl", "version 2"],
    "BSD-3-Clause": ["bsd 3-clause", "redistribution and use in source and binary"],
    "BSD-2-Clause": ["bsd 2-clause", "redistribution and use in source and binary forms"],
    "LGPL-3.0": ["gnu lesser general public license", "lgpl"],
    "MPL-2.0": ["mozilla public license", "mpl"],
    "ISC": ["isc license", "isc"],
    "CC-BY-4.0": ["creative commons attribution 4.0", "cc-by"],
    "CC-BY-SA-4.0": ["creative commons attribution-sharealike 4.0", "cc-by-sa"],
    "Unlicense": ["unlicense", "public domain"],
}

PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),
    ("phone_cn", re.compile(r"1[3-9]\d{9}")),
    ("phone_intl", re.compile(r"\+\d{1,3}[\s\-]?\d{3,4}[\s\-]?\d{4,}")),
    ("phone_us", re.compile(r"\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}")),
    ("id_card_cn", re.compile(r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]")),
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]

_PHONE_PLACEHOLDER = "[PHONE]"

PII_PLACEHOLDERS: dict[str, str] = {
    "email": "[EMAIL]",
    "phone_cn": _PHONE_PLACEHOLDER,
    "phone_intl": _PHONE_PLACEHOLDER,
    "phone_us": _PHONE_PLACEHOLDER,
    "id_card_cn": "[ID_CARD]",
    "ipv4": "[IP_ADDRESS]",
}


class MetadataManager:
    def __init__(self):
        self._version_map: dict[str, dict] = {}

    def redact_pii(self, text: str) -> tuple[str, list[dict]]:
        findings: list[dict] = []
        redacted = text
        for pii_type, pattern in PII_PATTERNS:
            for match in pattern.finditer(text):
                findings.append({"type": pii_type, "value": match.group(), "start": match.start(), "end": match.end()})
            redacted = pattern.sub(PII_PLACEHOLDERS.get(pii_type, "[REDACTED]"), redacted)
        return redacted, findings

    def redact_pii_metadata(self, metadata: dict) -> dict:
        result = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                redacted, _ = self.redact_pii(value)
                result[key] = redacted
            elif isinstance(value, dict):
                result[key] = self.redact_pii_metadata(value)
            elif isinstance(value, list):
                result[key] = [
                    self.redact_pii_metadata(item) if isinstance(item, dict)
                    else (self.redact_pii(item)[0] if isinstance(item, str) else item)
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def detect_license(self, text: str) -> str | None:
        text_lower = text.lower()
        best_match = None
        best_score = 0
        for license_name, patterns in LICENSE_PATTERNS.items():
            score = sum(1 for p in patterns if p in text_lower)
            if score > best_score:
                best_score = score
                best_match = license_name
        if best_score >= 2:
            return best_match
        if best_score == 1 and best_match in ("MIT", "ISC", "Unlicense"):
            return best_match
        return None

    def add_copyright_metadata(self, metadata: dict, source: str, license_type: str | None = None) -> dict:
        result = dict(metadata)
        result["copyright_source"] = source
        if license_type:
            result["copyright_license"] = license_type
        return result

    def process_metadata(self, metadata: dict) -> dict:
        result = self.redact_pii_metadata(metadata)
        if "source" in result:
            source = result["source"]
            license_type = result.get("license") or result.get("copyright_license")
            if license_type is None and isinstance(source, str):
                license_type = self.detect_license(source)
            result = self.add_copyright_metadata(result, source, license_type)
        return result

    def get_version(self, doc_hash: str) -> dict | None:
        return self._version_map.get(doc_hash)

    def set_version(self, doc_hash: str, record: dict) -> None:
        self._version_map[doc_hash] = record

    def remove_version(self, doc_hash: str) -> None:
        self._version_map.pop(doc_hash, None)

    def list_versions(self) -> dict[str, dict]:
        return dict(self._version_map)
