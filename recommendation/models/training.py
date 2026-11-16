"""XGBoost training with three cross-validation schemes.

CV schemes:
- Grouped k-Fold (5 folds, grouped by run_id — same run never leaks across folds)
- LOCO: Leave-One-Config-Out (5 folds, one executor config held out each time)
- LOSO: Leave-One-SF-Out (5 folds, one scale_factor held out each time)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from recommendation.config import CONSOLIDATED_DIR, EXECUTOR_CONFIGS, MODELS_DIR

logger = logging.getLogger(__name__)

# ── Feature families ─────────────────────────────────────────────────────────

# Core features: always available (100% non-null in success rows across all exps)
# aqe_enabled removed: zero variance (all 13,998 success rows have AQE=True → SHAP≈0)
CORE_FEATURES = [
    "scale_factor",
    "workers",
    "executor_cores",
    "executor_memory_gb",
    "total_cores",
    "total_memory_gb",
    "cores_per_gb_data",
    "memory_per_gb_data",
    "data_per_worker_gb",
    "query_category_enc",  # label-encoded from query_category
]

# Plan features: available only in Exp05 (~16% of success rows)
PLAN_FEATURES = [
    "num_joins",
    "num_broadcast_joins",
    "num_sort_merge_joins",
    "num_exchanges",
    "num_sorts",
    "num_aggregates",
    "num_filters",
    "num_scans",
    "num_subqueries",
    "num_windows",
    "num_unions",
    "plan_depth",
    "plan_lines",
    "plan_complexity",
]

CONFIG_MAP = {
    (2, 7.0): "xsmall",
    (3, 7.0): "xsmall",
    (4, 14.0): "small",
    (8, 28.0): "medium",
    (12, 48.0): "large",
    (24, 96.0): "xlarge",
}

CONFIG_ORDER = ["xsmall", "small", "medium", "large", "xlarge"]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    # y_true / y_pred are in log1p(ms) space
    t_true_ms = np.expm1(y_true)
    t_pred_ms = np.expm1(y_pred)
    mape_original = float(np.mean(np.abs(t_true_ms - t_pred_ms) / np.maximum(t_true_ms, 1e-6)))
    rmse_original_s = float(np.sqrt(np.mean(((t_true_ms - t_pred_ms) / 1000) ** 2)))
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "rmse": _rmse(y_true, y_pred),          # log1p space — training metric
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape_log": float(mean_absolute_percentage_error(y_true, y_pred)),  # log space only
        "mape_original": mape_original,          # |t_pred-t_true|/t_true after expm1
        "rmse_original_s": rmse_original_s,      # RMSE in seconds
        "r2": float(r2_score(y_true, y_pred)),
        "spearman_rho": float(rho),              # ranking quality per fold
    }


def load_and_prepare(parquet_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load parquet, filter success, engineer features.

    Returns (X, y, groups) where groups is run_id encoded as int.
    """
    df = pd.read_parquet(parquet_path)
    df = df[df["status"] == "success"].copy()
    logger.info("Success rows: %d", len(df))

    # config_name derived from executor shape
    df["config_name"] = df.apply(
        lambda r: CONFIG_MAP.get((r["executor_cores"], r["executor_memory_gb"]), "unknown"),
        axis=1,
    )

    # Label-encode query_category
    le = LabelEncoder()
    df["query_category_enc"] = le.fit_transform(df["query_category"])

    # Check plan feature availability
    plan_null_frac = df[PLAN_FEATURES].isna().mean().mean()
    logger.info("Plan features mean NaN fraction: %.1f%%", plan_null_frac * 100)

    # Use plan features only if <50% NaN (threshold: if majority is usable)
    use_plan = plan_null_frac < 0.50
    feature_cols = CORE_FEATURES + (PLAN_FEATURES if use_plan else [])

    if not use_plan:
        logger.warning(
            "Plan features discarded: %.1f%% NaN — only core features used",
            plan_null_frac * 100,
        )

    df["_features_used_plan"] = use_plan

    # Fill plan NaN with -1 sentinel if they're included (XGBoost handles it natively)
    if use_plan:
        df[PLAN_FEATURES] = df[PLAN_FEATURES].fillna(-1)

    X = df[feature_cols].copy()
    y = np.log1p(df["duration_ms"].values)  # log1p target for MAPE stability

    # Groups: run_id → int (for GroupKFold)
    run_ids = df["run_id"].astype("category").cat.codes.values

    # Store metadata columns alongside X for CV splitting
    X["_scale_factor"] = df["scale_factor"].values
    X["_config_name"] = df["config_name"].values
    X["_run_id_code"] = run_ids
    X["_query"] = df["query"].values

    return X, y, run_ids, df, le, use_plan, feature_cols


def _xgb_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="rmse",
        verbosity=0,
    )


def cv_grouped_kfold(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int = 5
) -> Dict:
    """Grouped k-Fold: groups by run_id so no run leaks across train/val."""
    feature_cols = [c for c in X.columns if not c.startswith("_")]
    X_feat = X[feature_cols]

    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X_feat, y, groups)):
        X_tr, X_val = X_feat.iloc[train_idx], X_feat.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = _xgb_model()
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_pred = model.predict(X_val)
        metrics = _compute_metrics(y_val, y_pred)
        metrics["fold"] = fold_i
        metrics["val_size"] = len(val_idx)

        # Spearman ρ per SF within this fold (ranking quality per scale)
        sfs_val = X["_scale_factor"].iloc[val_idx].values
        spearman_per_sf = {}
        for sf in sorted(np.unique(sfs_val)):
            sf_mask = sfs_val == sf
            if sf_mask.sum() >= 3:
                rho, _ = spearmanr(y_val[sf_mask], y_pred[sf_mask])
                spearman_per_sf[int(sf)] = float(rho)
        metrics["spearman_per_sf"] = spearman_per_sf

        fold_metrics.append(metrics)
        logger.info("Grouped k-Fold fold %d: RMSE=%.4f R2=%.4f rho=%.4f", fold_i, metrics["rmse"], metrics["r2"], metrics["spearman_rho"])

    return {"per_fold": fold_metrics}


def cv_loqo(X: pd.DataFrame, y: np.ndarray, queries: np.ndarray, n_splits: int = 11) -> Dict:
    """Leave-Queries-Out: dobras por consulta, nunca por célula.

    Os outros esquemas deixam a consulta sempre no treino; este mede o caso
    do usuário do framework, que chega com uma consulta que o modelo nunca
    viu e só pode descrevê-la pelas contagens do plano físico. 11 dobras de 9
    consultas (99 ao todo); cada dobra retira a consulta de todas as células.
    """
    feature_cols = [c for c in X.columns if not c.startswith("_")]
    X_feat = X[feature_cols]
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X_feat, y, queries)):
        X_tr, X_val = X_feat.iloc[train_idx], X_feat.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = _xgb_model()
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        metrics = _compute_metrics(y_val, y_pred)
        metrics["fold"] = fold_i
        metrics["val_size"] = len(val_idx)
        metrics["held_out_queries"] = sorted(str(q) for q in np.unique(queries[val_idx]))
        fold_metrics.append(metrics)
        logger.info("LOQO fold %d (%s): RMSE=%.4f R2=%.4f rho=%.4f", fold_i,
                    ",".join(metrics["held_out_queries"]), metrics["rmse"], metrics["r2"], metrics["spearman_rho"])
    return {"per_fold": fold_metrics}


def cv_loco(X: pd.DataFrame, y: np.ndarray) -> Dict:
    """Leave-One-Config-Out: hold out one executor config family at a time."""
    feature_cols = [c for c in X.columns if not c.startswith("_")]
    X_feat = X[feature_cols]
    configs = X["_config_name"].values
    unique_configs = [c for c in CONFIG_ORDER if c in np.unique(configs)]

    fold_metrics = []
    for cfg in unique_configs:
        val_mask = configs == cfg
        train_mask = ~val_mask

        if val_mask.sum() < 10:
            logger.warning("LOCO fold %s: only %d val rows, skipping", cfg, val_mask.sum())
            continue

        X_tr, X_val = X_feat[train_mask], X_feat[val_mask]
        y_tr, y_val = y[train_mask], y[val_mask]

        model = _xgb_model()
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_pred = model.predict(X_val)
        metrics = _compute_metrics(y_val, y_pred)
        metrics["fold_config"] = cfg
        metrics["val_size"] = int(val_mask.sum())
        fold_metrics.append(metrics)
        logger.info("LOCO fold %s: RMSE=%.4f R2=%.4f", cfg, metrics["rmse"], metrics["r2"])

    return {"per_fold": fold_metrics}


def cv_loso(X: pd.DataFrame, y: np.ndarray) -> Dict:
    """Leave-One-SF-Out: hold out one scale_factor at a time.

    Per-fold metrics reported separately (not averaged) because SF=10 vs SF=500
    represent qualitatively different extrapolation difficulty.
    """
    feature_cols = [c for c in X.columns if not c.startswith("_")]
    X_feat = X[feature_cols]
    sfs = X["_scale_factor"].values
    unique_sfs = sorted(np.unique(sfs))

    fold_metrics = []
    for sf in unique_sfs:
        val_mask = sfs == sf
        train_mask = ~val_mask

        if val_mask.sum() < 10:
            logger.warning("LOSO fold SF=%s: only %d val rows, skipping", sf, val_mask.sum())
            continue

        X_tr, X_val = X_feat[train_mask], X_feat[val_mask]
        y_tr, y_val = y[train_mask], y[val_mask]

        model = _xgb_model()
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_pred = model.predict(X_val)
        metrics = _compute_metrics(y_val, y_pred)
        metrics["fold_sf"] = int(sf)
        metrics["val_size"] = int(val_mask.sum())
        fold_metrics.append(metrics)
        logger.info("LOSO fold SF=%d: RMSE=%.4f R2=%.4f", sf, metrics["rmse"], metrics["r2"])

    return {"per_fold": fold_metrics}


def train_final_model(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray
) -> XGBRegressor:
    """Train final model on all data (for SHAP and production use)."""
    feature_cols = [c for c in X.columns if not c.startswith("_")]
    X_feat = X[feature_cols]

    # Use last GroupKFold split as eval set for early stopping
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_feat, y, groups))
    _, val_idx = splits[-1]
    train_idx = np.concatenate([tr for tr, _ in splits[:-1]])

    model = _xgb_model()
    model.fit(
        X_feat.iloc[train_idx], y[train_idx],
        eval_set=[(X_feat.iloc[val_idx], y[val_idx])],
        verbose=False,
    )
    # Refit on full data with best n_estimators (no early stopping)
    best_n = model.best_iteration + 1
    final = XGBRegressor(
        n_estimators=best_n,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    final.fit(X_feat, y)
    logger.info("Final model trained: n_estimators=%d", best_n)
    return final


def run_training(
    parquet_path: Path = CONSOLIDATED_DIR / "training_data.parquet",
    models_dir: Path = MODELS_DIR,
) -> Dict:
    """Full training pipeline: load → CV × 3 → final model → save artefacts."""
    models_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups, df, le, use_plan, feature_cols = load_and_prepare(parquet_path)

    cv_results: Dict = {}

    logger.info("Running Grouped k-Fold CV...")
    cv_results["xgboost"] = {
        "grouped_kfold": cv_grouped_kfold(X, y, groups),
        "loco": cv_loco(X, y),
        "loso": cv_loso(X, y),
        "loqo": cv_loqo(X, y, X["_query"].values),
        "_meta": {
            "features": feature_cols,
            "use_plan_features": use_plan,
            "n_train": len(y),
            "target": "log1p(duration_ms)",
        },
    }

    # Save CV results
    cv_path = models_dir / "cv_results.json"
    with open(cv_path, "w") as f:
        json.dump(cv_results, f, indent=2, default=str)
    logger.info("CV results saved: %s", cv_path)

    # Train and save final model
    logger.info("Training final model...")
    final_model = train_final_model(X, y, groups)
    model_path = models_dir / "xgb_gold.joblib"
    joblib.dump(
        {
            "model": final_model,
            "feature_cols": feature_cols,
            "label_encoder": le,
            "use_plan_features": use_plan,
            "config_map": CONFIG_MAP,
        },
        model_path,
    )
    logger.info("Model saved: %s", model_path)

    return cv_results, final_model, X, y, feature_cols
