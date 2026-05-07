"""Shared metric utilities for the eval suite."""
import math
import re


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if relevant & set(retrieved[:k]) else 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    gains = [1.0 if r in relevant else 0.0 for r in retrieved[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def check_expected_value(rows: list[dict], expected: dict | None) -> bool:
    if expected is None:
        return True
    op, val = expected["op"], expected["value"]
    if not rows:
        return False
    actual = list(rows[0].values())[0]
    if actual is None:
        return False
    if op == "contains":
        return str(val).lower() in str(actual).lower()
    elif op == "=":
        return abs(float(actual) - float(val)) < 1e-6
    elif op == ">=":
        return float(actual) >= float(val)
    elif op == "<=":
        return float(actual) <= float(val)
    elif op == "range":
        lo, hi = val[0], val[1]
        return lo <= float(actual) <= hi
    return False
