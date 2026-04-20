from __future__ import annotations

import math
from typing import Sequence

_EPSILON = 1e-10

from pydantic import BaseModel


class QueryEvaluation(BaseModel):
    query: str
    relevant_ids: set[str]
    retrieved_ids: list[str]


class RetrievalEvaluationResult(BaseModel):
    mrr: float
    ndcg_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    hit_rate: float
    num_queries: int


def reciprocal_rank(relevant_ids: set[str], retrieved_ids: Sequence[str]) -> float:
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def mrr(evaluations: Sequence[QueryEvaluation]) -> float:
    if not evaluations:
        return 0.0
    total = sum(reciprocal_rank(e.relevant_ids, e.retrieved_ids) for e in evaluations)
    return total / len(evaluations)


def dcg_at_k(relevant_ids: set[str], retrieved_ids: Sequence[str], k: int) -> float:
    score = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            score += 1.0 / math.log2(i + 2)
    return score


def idcg_at_k(num_relevant: int, k: int) -> float:
    score = 0.0
    for i in range(min(num_relevant, k)):
        score += 1.0 / math.log2(i + 2)
    return score


def ndcg_at_k(relevant_ids: set[str], retrieved_ids: Sequence[str], k: int) -> float:
    ideal = idcg_at_k(len(relevant_ids), k)
    if ideal < _EPSILON:
        return 0.0
    actual = dcg_at_k(relevant_ids, retrieved_ids, k)
    return actual / ideal


def recall_at_k(relevant_ids: set[str], retrieved_ids: Sequence[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    return len(relevant_ids & retrieved_set) / len(relevant_ids)


def precision_at_k(relevant_ids: set[str], retrieved_ids: Sequence[str], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    return len(relevant_ids & retrieved_set) / k


def hit_rate(evaluations: Sequence[QueryEvaluation]) -> float:
    if not evaluations:
        return 0.0
    hits = sum(1 for e in evaluations if e.relevant_ids & set(e.retrieved_ids))
    return hits / len(evaluations)


def evaluate(
    evaluations: Sequence[QueryEvaluation],
    ks: Sequence[int] = (5, 10, 20),
) -> RetrievalEvaluationResult:
    ndcg_scores: dict[int, float] = {}
    recall_scores: dict[int, float] = {}
    precision_scores: dict[int, float] = {}

    for k in ks:
        ndcg_values = [ndcg_at_k(e.relevant_ids, e.retrieved_ids, k) for e in evaluations]
        recall_values = [recall_at_k(e.relevant_ids, e.retrieved_ids, k) for e in evaluations]
        precision_values = [precision_at_k(e.relevant_ids, e.retrieved_ids, k) for e in evaluations]

        ndcg_scores[k] = sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0
        recall_scores[k] = sum(recall_values) / len(recall_values) if recall_values else 0.0
        precision_scores[k] = sum(precision_values) / len(precision_values) if precision_values else 0.0

    return RetrievalEvaluationResult(
        mrr=mrr(evaluations),
        ndcg_at_k=ndcg_scores,
        recall_at_k=recall_scores,
        precision_at_k=precision_scores,
        hit_rate=hit_rate(evaluations),
        num_queries=len(evaluations),
    )
