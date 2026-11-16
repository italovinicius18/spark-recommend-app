"""Baseline models for comparison against the recommendation framework.

Literature justification:
- CherryPick (NSDI 2017): GP + EI for cloud configuration
- LOCAT (SIGMOD 2022): BO with data-aware GP
- Rover (KDD 2023): Transfer learning + BO
- Li et al. (VLDB 2023): Generalized multi-goal BO

These baselines answer the reviewer question: "Why XGBoost instead of
Bayesian Optimization?"  Answer: BO excels with <100 observations for
online exploration. With 13K+ historical data points, supervised models
(XGBoost, RF) leverage the full dataset more effectively.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DefaultConfigBaseline:
    """Baseline: always recommend Spark defaults + AQE.

    Represents the "do nothing" scenario. If our framework can't beat
    this, it has no value.
    """

    def __init__(self, default_config: str = "small", default_workers: int = 4):
        self.config = default_config
        self.workers = default_workers

    def fit(self, X: pd.DataFrame, y: np.ndarray, config_col: str = "config_name",
            workers_col: str = "workers"):
        mask = (X[config_col] == self.config) & (X[workers_col] == self.workers)
        self._default_duration = float(y[mask].mean()) if mask.any() else float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._default_duration)


class CategoryMedianBaseline:
    """Baseline: predict median duration per query category.

    Tests whether query-level features add value beyond simple category
    grouping.
    """

    def __init__(self, category_col: str = "query_category"):
        self.category_col = category_col
        self._medians: Dict[str, float] = {}
        self._global_median: float = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        df = X.copy()
        df["_target"] = y
        self._medians = df.groupby(self.category_col)["_target"].median().to_dict()
        self._global_median = float(np.median(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([
            self._medians.get(row[self.category_col], self._global_median)
            for _, row in X.iterrows()
        ])


class RandomConfigBaseline:
    """Baseline: random valid configuration from the search space.

    Lower bound — any reasonable method should beat this.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._mean_per_config: Dict[Tuple, float] = {}
        self._global_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray,
            config_col: str = "config_name", workers_col: str = "workers"):
        df = X.copy()
        df["_target"] = y
        for (cfg, w), grp in df.groupby([config_col, workers_col]):
            self._mean_per_config[(cfg, w)] = float(grp["_target"].mean())
        self._global_mean = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        configs = list(self._mean_per_config.values())
        if not configs:
            return np.full(len(X), self._global_mean)
        indices = self.rng.randint(0, len(configs), size=len(X))
        return np.array([configs[i] for i in indices])


class BayesianOptimizationBaseline:
    """Baseline: Gaussian Process regression (CherryPick-style, batch variant).

    Trains a GP on the full dataset (subsampled for tractability) and
    predicts duration directly.  This mirrors what CherryPick does
    internally — it is the GP *model*, not the exploration loop.
    Comparison against XGBoost isolates model capacity from exploration
    strategy.
    """

    def __init__(self, n_restarts: int = 5, alpha: float = 1e-6,
                 max_train_samples: int = 2000):
        self.n_restarts = n_restarts
        self.alpha = alpha
        self.max_train_samples = max_train_samples
        self._model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        X_num = X.select_dtypes(include=[np.number])
        self._num_cols = X_num.columns.tolist()
        X_scaled = self._scaler.fit_transform(X_num)

        rng = np.random.RandomState(42)
        if len(X_scaled) > self.max_train_samples:
            logger.info("GP-batch: subsampling %d/%d for tractability",
                        self.max_train_samples, len(X_scaled))
            idx = rng.choice(len(X_scaled), self.max_train_samples, replace=False)
            X_scaled = X_scaled[idx]
            y = y[idx]

        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1.0)
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            alpha=self.alpha,
            random_state=42,
        )
        self._model.fit(X_scaled, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self._scaler.transform(X[self._num_cols])
        return self._model.predict(X_scaled)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) for use in acquisition functions."""
        X_scaled = self._scaler.transform(X[self._num_cols])
        return self._model.predict(X_scaled, return_std=True)


class BayesianOptimizationIterativeBaseline:
    """Baseline: iterative BO with Expected Improvement (CherryPick online loop).

    Simulates the CherryPick/LOCAT online configuration search:
    start with a small random seed, then iteratively select the next
    candidate using EI acquisition and query its true cost.

    After n_iterations BO steps the best-seen configuration is returned.
    For prediction on test rows, the BO loop is run independently for
    each (scale_factor, query_category) group using pool rows from the
    training set.  This mirrors real-world online search where you explore
    configs for a specific workload type before committing.

    This is the scientifically correct CherryPick comparison — not just
    a batch GP fit.  It exposes BO's sample-efficiency advantage *and*
    its ceiling when a large offline dataset is already available.
    """

    def __init__(
        self,
        n_seed: int = 5,
        n_iterations: int = 30,
        n_restarts_gp: int = 3,
        alpha: float = 1e-4,
        xi: float = 0.01,
        seed: int = 42,
    ):
        self.n_seed = n_seed
        self.n_iterations = n_iterations
        self.n_restarts_gp = n_restarts_gp
        self.alpha = alpha
        self.xi = xi
        self.seed = seed
        self._best_per_group: Dict[Tuple, float] = {}
        self._global_mean: float = 0.0

    @staticmethod
    def _expected_improvement(
        X_candidates: np.ndarray,
        gp,
        y_best: float,
        xi: float = 0.01,
    ) -> np.ndarray:
        from scipy.stats import norm

        mu, sigma = gp.predict(X_candidates, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        # minimising duration → improvement when mu < y_best
        z = (y_best - mu - xi) / sigma
        ei = (y_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-9] = 0.0
        return ei

    def _run_bo_loop(
        self,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        rng: np.random.RandomState,
    ) -> float:
        """Run BO loop on a candidate pool; return best observed y."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel

        n = len(X_pool)
        if n <= self.n_seed:
            return float(y_pool.min())

        observed_idx = set(rng.choice(n, min(self.n_seed, n), replace=False).tolist())

        for _ in range(self.n_iterations):
            obs = list(observed_idx)
            X_obs = X_pool[obs]
            y_obs = y_pool[obs]

            kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-2)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=self.n_restarts_gp,
                alpha=self.alpha,
                random_state=42,
            )
            try:
                gp.fit(X_obs, y_obs)
            except Exception:
                break

            unobserved = [i for i in range(n) if i not in observed_idx]
            if not unobserved:
                break

            X_cand = X_pool[unobserved]
            ei = self._expected_improvement(X_cand, gp, float(y_obs.min()), self.xi)
            best_local = int(np.argmax(ei))
            observed_idx.add(unobserved[best_local])

        return float(y_pool[list(observed_idx)].min())

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sf_col: str = "_scale_factor",
        cat_col: str = "query_category_enc",
    ):
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        X_num = X.select_dtypes(include=[np.number])
        self._num_cols = X_num.columns.tolist()
        X_scaled = self._scaler.fit_transform(X_num)

        rng = np.random.RandomState(self.seed)
        self._global_mean = float(y.mean())

        sf_arr = X[sf_col].values if sf_col in X.columns else np.zeros(len(X))
        cat_arr = X[cat_col].values if cat_col in X.columns else np.zeros(len(X))

        unique_groups = np.unique(np.column_stack([sf_arr, cat_arr]), axis=0)
        for sf, cat in unique_groups:
            mask = (sf_arr == sf) & (cat_arr == cat)
            if mask.sum() < 3:
                continue
            best = self._run_bo_loop(X_scaled[mask], y[mask], rng)
            self._best_per_group[(float(sf), float(cat))] = best
            logger.debug("BO-iterative SF=%g cat=%g best=%.4f", sf, cat, best)

        logger.info("BO-iterative fit done: %d groups", len(self._best_per_group))
        return self

    def predict(
        self,
        X: pd.DataFrame,
        sf_col: str = "_scale_factor",
        cat_col: str = "query_category_enc",
    ) -> np.ndarray:
        sf_arr = X[sf_col].values if sf_col in X.columns else np.zeros(len(X))
        cat_arr = X[cat_col].values if cat_col in X.columns else np.zeros(len(X))
        return np.array([
            self._best_per_group.get((float(sf), float(cat)), self._global_mean)
            for sf, cat in zip(sf_arr, cat_arr)
        ])


# Registry for easy access
BASELINE_REGISTRY = {
    "default_config": DefaultConfigBaseline,
    "random_config": RandomConfigBaseline,
    "category_median": CategoryMedianBaseline,
    "bayesian_optimization": BayesianOptimizationBaseline,
    "bayesian_optimization_iterative": BayesianOptimizationIterativeBaseline,
}
