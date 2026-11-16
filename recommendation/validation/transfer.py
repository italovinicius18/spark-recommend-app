"""Cross-workload transfer validation (Paper A / UCC 2026).

Scores the frozen TPC-DS-trained gold model on TPC-H records:
zero-shot prediction error (RQ1) and recommendation regret vs the
per-SF oracle over the collected cell grid (RQ4).
"""

import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from recommendation.config import MODELS_DIR
from recommendation.data.schema import QueryExecution, dataclass_to_dataframe
from recommendation.models.training import (
    CONFIG_MAP,
    PLAN_FEATURES,
    _compute_metrics,
)

logger = logging.getLogger(__name__)

# Spark-defaults-shaped baseline cell: small executors (4c/14g), 4 workers.
DEFAULT_BASELINE_CONFIG = ("small", 4)

_REQUIRED_BUNDLE_KEYS = {"model", "feature_cols", "label_encoder", "use_plan_features"}


def load_model_bundle(path: Path = MODELS_DIR / "xgb_gold.joblib") -> Dict:
    bundle = joblib.load(path)
    missing = _REQUIRED_BUNDLE_KEYS - set(bundle)
    if missing:
        raise ValueError(f"model bundle at {path} missing keys: {sorted(missing)}")
    return bundle


def build_feature_frame(
    records: Iterable[QueryExecution], bundle: Dict
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Reproduce the gold model's training-time feature engineering.

    Uses the *saved* label encoder (transform, never fit) so category codes
    match training. Returns (X[feature_cols], y_log1p, meta_df).
    """
    df = dataclass_to_dataframe(list(records))
    if df.empty:
        raise ValueError("no records to score")
    df = df[df["status"] == "success"].copy()
    if df.empty:
        raise ValueError("no success records to score")

    df["config_name"] = df.apply(
        lambda r: CONFIG_MAP.get(
            (r["executor_cores"], r["executor_memory_gb"]), "unknown"
        ),
        axis=1,
    )
    df["query_category_enc"] = bundle["label_encoder"].transform(df["query_category"])

    if bundle["use_plan_features"]:
        present = [c for c in PLAN_FEATURES if c in df.columns]
        df[present] = df[present].fillna(-1)

    X = df[bundle["feature_cols"]].copy()
    y = np.log1p(df["duration_ms"].values)
    return X, y, df.reset_index(drop=True)


def zero_shot_report(bundle: Dict, records: Iterable[QueryExecution]) -> Dict:
    """RQ1: prediction quality of the frozen model on a foreign workload."""
    X, y, meta = build_feature_frame(records, bundle)
    y_pred = np.asarray(bundle["model"].predict(X))

    report = {
        "n": int(len(y)),
        "overall": _compute_metrics(y, y_pred),
        "per_sf": {},
        "per_category": {},
    }
    for sf in sorted(meta["scale_factor"].unique()):
        mask = (meta["scale_factor"] == sf).values
        if mask.sum() >= 3:
            report["per_sf"][int(sf)] = _compute_metrics(y[mask], y_pred[mask])
    for cat in sorted(meta["query_category"].unique()):
        mask = (meta["query_category"] == cat).values
        if mask.sum() >= 3:
            report["per_category"][cat] = _compute_metrics(y[mask], y_pred[mask])
    return report


def recommendation_regret(
    bundle: Dict, records: Iterable[QueryExecution], min_coverage: float = 0.9
) -> Dict:
    """RQ4: per-SF regret of the predicted-best cell vs the observed oracle.

    Cell = (config_name, workers). Cell score = sum over queries of the
    median duration across tries (actual and predicted alike).

    Two-stage fairness guard (success-only sums would otherwise make a cell
    that crashed its heavy queries look "fast" — observed with xsmall cells
    OOMing heavy TPC-H queries while completing the light ones):
    1. Cells whose query coverage < min_coverage of the SF's query set are
       excluded as *infeasible* (reported in `excluded_cells` — a finding,
       not noise: the frozen model cannot predict failure).
    2. Surviving cells are compared on the intersection of queries that
       succeeded in every one of them.
    """
    X, y, meta = build_feature_frame(records, bundle)
    meta = meta.copy()
    meta["pred_ms"] = np.expm1(np.asarray(bundle["model"].predict(X)))
    meta["actual_ms"] = meta["duration_ms"]

    per_query = (
        meta.groupby(["scale_factor", "config_name", "workers", "query"])[
            ["actual_ms", "pred_ms"]
        ]
        .median()
        .reset_index()
    )

    kept = []
    common_queries: Dict = {}
    excluded_cells: Dict = {}
    for sf, grp in per_query.groupby("scale_factor"):
        q_total = grp["query"].nunique()
        cov = grp.groupby(["config_name", "workers"])["query"].nunique()
        bad = cov[cov < min_coverage * q_total]
        excluded_cells[int(sf)] = [
            {"config": c, "workers": int(w), "coverage": round(n / q_total, 2)}
            for (c, w), n in bad.items()
        ]
        good_keys = set(cov[cov >= min_coverage * q_total].index)
        grp = grp[
            grp.set_index(["config_name", "workers"]).index.isin(good_keys)
        ]
        n_cells = grp.groupby(["config_name", "workers"]).ngroups
        counts = grp.groupby("query")["config_name"].count()
        common = set(counts[counts == n_cells].index)
        common_queries[int(sf)] = sorted(common)
        kept.append(grp[grp["query"].isin(common)])
    per_query = pd.concat(kept, ignore_index=True)

    per_cell = (
        per_query.groupby(["scale_factor", "config_name", "workers"])[
            ["actual_ms", "pred_ms"]
        ]
        .sum()
        .reset_index()
    )

    out: Dict = {}
    for sf, grp in per_cell.groupby("scale_factor"):
        oracle = grp.loc[grp["actual_ms"].idxmin()]
        rec = grp.loc[grp["pred_ms"].idxmin()]
        entry = {
            "oracle": {
                "config": oracle["config_name"],
                "workers": int(oracle["workers"]),
                "runtime_s": float(oracle["actual_ms"]) / 1000.0,
            },
            "recommended": {
                "config": rec["config_name"],
                "workers": int(rec["workers"]),
                "runtime_s": float(rec["actual_ms"]) / 1000.0,
            },
            "regret_pct": float(
                (rec["actual_ms"] - oracle["actual_ms"]) / oracle["actual_ms"] * 100.0
            ),
            "n_cells": int(len(grp)),
            "n_common_queries": len(common_queries[int(sf)]),
            "excluded_cells": excluded_cells[int(sf)],
        }
        default = grp[
            (grp["config_name"] == DEFAULT_BASELINE_CONFIG[0])
            & (grp["workers"] == DEFAULT_BASELINE_CONFIG[1])
        ]
        if not default.empty:
            entry["default_regret_pct"] = float(
                (default["actual_ms"].iloc[0] - oracle["actual_ms"])
                / oracle["actual_ms"]
                * 100.0
            )
        out[int(sf)] = entry
    return out


def _fit_frame(df: pd.DataFrame, feature_cols, encoder: LabelEncoder):
    df = df[df["status"] == "success"].copy()
    df["query_category_enc"] = encoder.transform(df["query_category"])
    plan_cols = [c for c in PLAN_FEATURES if c in feature_cols and c in df.columns]
    if plan_cols:
        df[plan_cols] = df[plan_cols].fillna(-1)
    return df[feature_cols], np.log1p(df["duration_ms"].values)


def _calibration_model() -> XGBRegressor:
    # Same shape as training._xgb_model but without early stopping (no eval
    # split needed here; mirrors train_final_model's refit stage).
    return XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=7, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0,
    )


def calibration_curve(
    train_parquet: Path,
    records: Iterable[QueryExecution],
    feature_cols,
    shots: tuple = (0, 1, 2, 4, 8, 16),
    seed: int = 42,
) -> Dict:
    """RQ2: few-shot — add N TPC-H run_ids to the TPC-DS training set,
    retrain, and score on the held-out TPC-H runs."""
    train_df = pd.read_parquet(train_parquet)
    tpch_df = dataclass_to_dataframe(list(records))
    tpch_df = tpch_df[tpch_df["status"] == "success"].copy()
    if tpch_df.empty:
        raise ValueError("no success TPC-H records for calibration")

    run_ids = sorted(tpch_df["run_id"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(run_ids)

    encoder = LabelEncoder()
    encoder.fit(
        pd.concat([train_df["query_category"], tpch_df["query_category"]]).unique()
    )

    curve: Dict = {}
    for n_shot in shots:
        if n_shot >= len(run_ids):
            logger.warning(
                "calibration: skipping shots=%d (only %d TPC-H runs)",
                n_shot, len(run_ids),
            )
            continue
        shot_ids = set(run_ids[:n_shot])
        eval_df = tpch_df[~tpch_df["run_id"].isin(shot_ids)]
        fit_df = (
            pd.concat(
                [train_df, tpch_df[tpch_df["run_id"].isin(shot_ids)]],
                ignore_index=True,
            )
            if n_shot
            else train_df
        )
        X_fit, y_fit = _fit_frame(fit_df, feature_cols, encoder)
        X_eval, y_eval = _fit_frame(eval_df, feature_cols, encoder)

        model = _calibration_model()
        model.fit(X_fit, y_fit)
        y_pred = model.predict(X_eval)
        curve[int(n_shot)] = {
            "n_eval": int(len(y_eval)),
            "n_shot_rows": int(len(fit_df) - len(train_df)),
            "overall": _compute_metrics(y_eval, y_pred),
        }
    return curve
