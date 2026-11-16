"""Deterministic query categories from physical-plan features.

The paper methodology requires categories derived from SQL/plan *structure*
via fixed thresholds — not hand labels, not observed runtimes (circularity
threat).

Calibration (2026-07-08, Exp 05 real EXPLAIN features, 99 queries): grid
search over (SUBQUERY_MIN 1-6, EXCHANGE_MIN 2-14, SMJ_MIN 1-8) maxes out at
53.5% agreement with the hand table (config.QUERY_CATEGORIES) at 5/5/2 — the
hand labels are NOT reproducible by a simple threshold rule over median plan
features. Analysis-phase decision for the paper: either regenerate categories
for BOTH benchmarks from this rule (non-circular, consistent), or keep hand
labels for both and describe the method as expert structural labeling. Do not
mix the two across benchmarks.
"""

from typing import Dict, Iterable, Optional

from recommendation.config import query_category
from recommendation.data.schema import QueryExecution, dataclass_to_dataframe

SUBQUERY_MIN = 5
EXCHANGE_MIN = 5
SMJ_MIN = 2

_PLAN_COLS = ["num_subqueries", "num_exchanges", "num_sort_merge_joins"]


def derive_category(
    num_subqueries: Optional[float],
    num_exchanges: Optional[float],
    num_sort_merge_joins: Optional[float],
) -> str:
    if (num_subqueries or 0) >= SUBQUERY_MIN:
        return "C_subquery"
    if (num_exchanges or 0) >= EXCHANGE_MIN or (num_sort_merge_joins or 0) >= SMJ_MIN:
        return "B_shuffle"
    return "A_scan"


def derive_categories(records: Iterable[QueryExecution]) -> Dict[str, str]:
    """Median plan features per query -> deterministic category."""
    df = dataclass_to_dataframe(list(records))
    if df.empty:
        return {}
    df = df.dropna(subset=_PLAN_COLS, how="all")
    if df.empty:
        return {}
    med = df.groupby("query")[_PLAN_COLS].median()
    return {
        q: derive_category(
            row["num_subqueries"], row["num_exchanges"], row["num_sort_merge_joins"]
        )
        for q, row in med.iterrows()
    }


def category_agreement(derived: Dict[str, str], benchmark: str) -> float:
    """Fraction of queries whose derived category matches the config table."""
    if not derived:
        return 0.0
    hits = sum(
        1 for q, cat in derived.items() if query_category(q, benchmark=benchmark) == cat
    )
    return hits / len(derived)
