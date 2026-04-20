from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from pydantic import BaseModel


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    vec_a = np.asarray(a, dtype=np.float64)
    vec_b = np.asarray(b, dtype=np.float64)
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if np.isclose(norm_a, 0.0) or np.isclose(norm_b, 0.0):
        return 0.0
    return float(dot / (norm_a * norm_b))


def cosine_similarity_batch(query: Sequence[float], candidates: Sequence[Sequence[float]]) -> list[float]:
    if not candidates:
        return []
    query_vec = np.asarray(query, dtype=np.float64)
    candidate_matrix = np.asarray(candidates, dtype=np.float64)
    query_norm = np.linalg.norm(query_vec)
    if np.isclose(query_norm, 0.0):
        return [0.0] * len(candidates)
    candidate_norms = np.linalg.norm(candidate_matrix, axis=1)
    dots = candidate_matrix @ query_vec
    denominators = candidate_norms * query_norm
    denominators = np.where(np.isclose(denominators, 0.0), 1.0, denominators)
    return (dots / denominators).tolist()


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


def jaccard_similarity_from_text(text_a: str, text_b: str) -> float:
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    return jaccard_similarity(tokens_a, tokens_b)


class LatencyStats(BaseModel):
    mean: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float
    count: int


def compute_latency_stats(latencies: Sequence[float]) -> LatencyStats:
    if not latencies:
        return LatencyStats(mean=0.0, p50=0.0, p95=0.0, p99=0.0, min=0.0, max=0.0, count=0)
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    mean_val = sum(sorted_latencies) / n

    def percentile(data: list[float], p: float) -> float:
        if n == 1:
            return data[0]
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        return data[int(f)] * (c - k) + data[int(c)] * (k - f)

    return LatencyStats(
        mean=mean_val,
        p50=percentile(sorted_latencies, 0.50),
        p95=percentile(sorted_latencies, 0.95),
        p99=percentile(sorted_latencies, 0.99),
        min=sorted_latencies[0],
        max=sorted_latencies[-1],
        count=n,
    )
