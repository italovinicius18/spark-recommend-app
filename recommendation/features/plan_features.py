"""Contagens estruturais de um `EXPLAIN EXTENDED` do Spark.

Cópia da função homônima de `scripts/run_tpcds_queries_v3.py`, que roda dentro
da imagem do benchmark e não enxerga este pacote. As duas precisam contar do
mesmo jeito: são as 14 features de plano do modelo. Se mudar aqui, mude lá.
"""
import re
from typing import Any, Dict

PLAN_FEATURE_KEYS = [
    "num_joins", "num_broadcast_joins", "num_sort_merge_joins", "num_exchanges",
    "num_sorts", "num_aggregates", "num_filters", "num_scans", "num_subqueries",
    "num_windows", "num_unions", "plan_depth", "plan_lines",
]


def physical_plan_section(plan_text: str) -> str:
    physical, inside = "", False
    for line in plan_text.splitlines():
        if "Physical Plan" in line:
            inside = True
            continue
        if inside:
            if line.startswith("==") and "==" in line[2:]:
                break
            physical += line + "\n"
    return physical or plan_text


def extract_plan_features(plan_text: str) -> Dict[str, Any]:
    plan = physical_plan_section(plan_text)
    count = lambda pat: len(re.findall(pat, plan, re.IGNORECASE))  # noqa: E731
    max_indent = max((len(l) - len(l.lstrip()) for l in plan.splitlines() if l.strip()), default=0)
    feats = {
        "num_joins": count(r"\b(BroadcastHashJoin|SortMergeJoin|ShuffledHashJoin|BroadcastNestedLoopJoin|CartesianProduct)\b"),
        "num_broadcast_joins": count(r"BroadcastHashJoin|BroadcastNestedLoopJoin"),
        "num_sort_merge_joins": count(r"SortMergeJoin"),
        "num_exchanges": count(r"\bExchange\b"),
        "num_sorts": count(r"\bSort\b"),
        "num_aggregates": count(r"\b(HashAggregate|SortAggregate|ObjectHashAggregate)\b"),
        "num_filters": count(r"\bFilter\b"),
        "num_scans": count(r"\b(FileScan|Scan)\b"),
        "num_subqueries": count(r"\bSubquery\b"),
        "num_windows": count(r"\bWindow\b"),
        "num_unions": count(r"\bUnion\b"),
        "plan_depth": max_indent,
        "plan_lines": len(plan.strip().splitlines()),
    }
    # Mesma fórmula do loader (`recommendation/data/loaders.py`).
    feats["plan_complexity"] = (feats["num_joins"] * 3 + feats["num_exchanges"] * 2
                                + feats["num_aggregates"] * 1.5 + feats["num_subqueries"] * 4
                                + feats["num_sorts"])
    return feats
