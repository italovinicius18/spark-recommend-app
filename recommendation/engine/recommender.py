"""Motor de recomendação: enumera candidatas, prevê tempo e custo, devolve a
fronteira de Pareto e a recomendação.

Entradas que o usuário tem de fato: o volume (GiB da Silver ou o fator de
escala nominal), a consulta (uma do catálogo medido ou as contagens de um
`EXPLAIN EXTENDED` colado), o envelope do cluster (nós com núcleos e memória)
e o objetivo. Saídas: candidatas viáveis com tempo e custo previstos,
fronteira, top-k com empates marcados, aviso de extrapolação e, quando a
consulta e a escala estão na matriz, a melhor célula medida para conferência.

Tudo que é número vem do mesmo lugar do capítulo 5: o bundle
`models/xgb_gold.joblib` treinado por `recommendation.models.training` sobre
`data/consolidated/training_data.parquet`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from recommendation.config import CONSOLIDATED_DIR, MODELS_DIR, SF_SIZE_GB
from recommendation.cost_model import run_cost_usd
from recommendation.features.plan_features import PLAN_FEATURE_KEYS
from recommendation.models.training import CONFIG_MAP, CONFIG_ORDER, PLAN_FEATURES

# Perfis de executor e contagens de workers da matriz (scripts/exp05_controller.py).
# O overhead entra na conta de viabilidade: o pod pede memória + overhead.
PROFILES: Dict[str, Dict] = {
    "xsmall": {"cores": 2, "memory_gb": 7.0, "overhead_gb": 1.0, "workers": [2, 4, 6, 8, 12, 16, 20]},
    "small": {"cores": 4, "memory_gb": 14.0, "overhead_gb": 2.0, "workers": [2, 4, 6, 8, 10, 12]},
    "medium": {"cores": 8, "memory_gb": 28.0, "overhead_gb": 4.0, "workers": [2, 4, 6, 8]},
    "large": {"cores": 12, "memory_gb": 48.0, "overhead_gb": 6.0, "workers": [2, 4, 6]},
    "xlarge": {"cores": 24, "memory_gb": 96.0, "overhead_gb": 8.0, "workers": [2, 4, 6]},
}

# Faixa de erro da previsão de tempo, fora da amostra, sobre o consolidado
# (scripts/check_early_stopping_bias.py, 21/09/2026): erro mediano 13,8%,
# 89% das previsões dentro de 1,5x. O piso de empate entre células é o CV
# entre replicações (scripts/replication_variance.py): 10%.
ERROR_MEDIAN = 0.138
WITHIN_1_5X = 0.89
TIE_TOLERANCE = 0.10

# Envelope medido: fatores de escala nominais da matriz.
MEASURED_SFS = sorted(k for k in SF_SIZE_GB if k >= 10)


@dataclass
class Node:
    cores: int
    memory_gb: float
    count: int = 1


@dataclass
class ClusterEnvelope:
    """Nós disponíveis para executores. O driver é desprezado (4c/8-16g).

    Cada nó reserva `reserve_cores`/`reserve_mem_gb` para os DaemonSets do
    sistema (kube-proxy, CNI), invisíveis a um acesso restrito ao namespace.
    Sem a reserva, um nó de 24 núcleos pareceria aceitar um executor XLarge
    de 24 núcleos cravados, e a matriz mostrou que não aceita.
    """
    nodes: List[Node]
    reserve_cores: int = 1
    reserve_mem_gb: float = 2.0

    @classmethod
    def parse(cls, spec: str) -> "ClusterEnvelope":
        """'5x48c187g,2x24c172g' -> dois grupos de nós."""
        nodes = []
        for part in spec.split(","):
            part = part.strip().lower()
            if not part:
                continue
            n, rest = (part.split("x", 1) + [None])[:2] if "x" in part else ("1", part)
            cores, mem = rest.rstrip("g").split("c")
            nodes.append(Node(int(cores), float(mem), int(n)))
        return cls(nodes)

    def executors_that_fit(self, profile: str) -> int:
        p = PROFILES[profile]
        need_mem = p["memory_gb"] + p["overhead_gb"]
        total = 0
        for nd in self.nodes:
            per_node = min((nd.cores - self.reserve_cores) // p["cores"],
                           int((nd.memory_gb - self.reserve_mem_gb) // need_mem))
            total += max(per_node, 0) * nd.count
        return total

    @classmethod
    def rnp(cls) -> "ClusterEnvelope":
        """Cluster da RNP em 2026-08 (CLAUDE.md, Node Capacity)."""
        return cls([Node(48, 187.0, 5), Node(24, 172.0, 2)])


@dataclass
class QuerySpec:
    """Consulta descrita pelas 14 contagens do plano e pela categoria."""
    name: str
    category: str
    plan: Dict[str, float]
    source: str = "catalogo"  # "catalogo" | "explain"


@dataclass
class Recommendation:
    candidates: pd.DataFrame
    top: pd.DataFrame
    best: pd.Series
    warnings: List[str] = field(default_factory=list)
    measured: Optional[pd.DataFrame] = None


@lru_cache(maxsize=1)
def load_bundle(path: Optional[str] = None) -> Dict:
    return joblib.load(Path(path) if path else MODELS_DIR / "xgb_gold.joblib")


@lru_cache(maxsize=1)
def load_catalog(parquet: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Catálogo das consultas medidas: por (consulta, SF) a categoria, as
    contagens de plano (moda) e, por célula, a mediana do tempo medido.

    As contagens variam com a escala em 57 das 99 consultas (o otimizador
    muda o plano com o tamanho das tabelas, caso q72), por isso a chave é
    (consulta, SF) e não só a consulta.
    """
    df = pd.read_parquet(Path(parquet) if parquet else CONSOLIDATED_DIR / "training_data.parquet")
    ok = df[df.status == "success"].copy()
    ok["profile"] = [CONFIG_MAP.get((c, m), "unknown") for c, m in zip(ok.executor_cores, ok.executor_memory_gb)]
    plans = (ok.groupby(["query", "scale_factor"])[["query_category"] + PLAN_FEATURES]
               .agg(lambda s: s.mode().iloc[0]).reset_index())
    cells = (ok.groupby(["query", "scale_factor", "profile", "workers"])
               .duration_ms.median().div(1000).rename("measured_s").reset_index())
    return {"plans": plans, "cells": cells, "queries": sorted(ok["query"].unique())}


def sf_equivalent(gib: float) -> float:
    """Volume em GiB -> fator de escala equivalente, interpolando a tabela de
    tamanhos da Silver em log-log. Fora da tabela extrapola pelo último par."""
    xs = np.array(sorted(SF_SIZE_GB)); ys = np.array([SF_SIZE_GB[k] for k in xs])
    return float(np.exp(np.interp(np.log(gib), np.log(ys), np.log(xs))))


def nearest_measured_sf(sf: float) -> int:
    return min(MEASURED_SFS, key=lambda s: abs(np.log(s) - np.log(max(sf, 1e-9))))


def query_from_catalog(name: str, sf: float, catalog: Dict[str, pd.DataFrame]) -> QuerySpec:
    p = catalog["plans"]
    sub = p[p["query"] == name]
    if sub.empty:
        raise KeyError(f"consulta {name} não está no catálogo")
    sf_near = sub.iloc[(sub.scale_factor - nearest_measured_sf(sf)).abs().argsort().iloc[0]]
    return QuerySpec(name, str(sf_near.query_category),
                     {k: float(sf_near[k]) for k in PLAN_FEATURES}, source="catalogo")


def query_from_explain(name: str, plan_text: str, category: str = "B_shuffle") -> QuerySpec:
    from recommendation.features.plan_features import extract_plan_features
    feats = extract_plan_features(plan_text)
    return QuerySpec(name, category, {k: float(feats[k]) for k in PLAN_FEATURES}, source="explain")


def query_from_sql(name: str, sql: str, sf: float, category: Optional[str] = None,
                   benchmark: str = "tpcds") -> QuerySpec:
    """Consulta escrita pelo usuário + magnitude do dado. O plano físico é
    gerado por uma sessão Spark local sobre o esquema com os tamanhos da
    magnitude (recommendation.features.local_explain), o que reproduz as
    decisões de broadcast e de filtro de Bloom do cluster."""
    from recommendation.features.local_explain import explain_sql, guess_category
    r = explain_sql(sql, sf, benchmark)
    feats = r["features"]
    cat = category or guess_category(feats)
    q = QuerySpec(name, cat, {k: float(feats[k]) for k in PLAN_FEATURES}, source="sql")
    q.plan_text = r["plan_text"]  # type: ignore[attr-defined]
    return q


def candidates(sf: float, envelope: ClusterEnvelope, data_gib: Optional[float] = None) -> pd.DataFrame:
    """Todas as (perfil, workers) da grade que cabem no envelope."""
    gib = data_gib if data_gib is not None else float(np.interp(sf, sorted(SF_SIZE_GB), [SF_SIZE_GB[k] for k in sorted(SF_SIZE_GB)]))
    rows = []
    for prof in CONFIG_ORDER:
        p = PROFILES[prof]
        fit = envelope.executors_that_fit(prof)
        for w in p["workers"]:
            if w > fit:
                continue
            tc, tm = p["cores"] * w, p["memory_gb"] * w
            rows.append({
                "profile": prof, "workers": w, "executor_cores": p["cores"],
                "executor_memory_gb": p["memory_gb"], "total_cores": tc, "total_memory_gb": tm,
                "scale_factor": sf, "cores_per_gb_data": tc / gib, "memory_per_gb_data": tm / gib,
                "data_per_worker_gb": gib / w,
            })
    return pd.DataFrame(rows)


def feature_frame(cands: pd.DataFrame, query: QuerySpec, bundle: Dict) -> pd.DataFrame:
    le = bundle["label_encoder"]
    cat_enc = int(le.transform([query.category])[0]) if query.category in set(le.classes_) else 0
    X = cands.copy()
    X["query_category_enc"] = cat_enc
    for k in PLAN_FEATURES:
        X[k] = query.plan.get(k, -1)
    cols = bundle["feature_cols"]
    return X[cols]


def pareto_mask(time_s: np.ndarray, cost: np.ndarray) -> np.ndarray:
    order = np.argsort(time_s)
    mask = np.zeros(len(time_s), dtype=bool)
    best = np.inf
    for i in order:
        if cost[i] < best:
            mask[i] = True
            best = cost[i]
    return mask


def recommend(query: QuerySpec, sf: float, envelope: ClusterEnvelope, objective: str = "time",
              top_k: int = 3, data_gib: Optional[float] = None, price: str = "ondemand",
              bundle: Optional[Dict] = None, catalog: Optional[Dict] = None) -> Recommendation:
    bundle = bundle or load_bundle()
    cands = candidates(sf, envelope, data_gib)
    warnings: List[str] = []
    if cands.empty:
        raise ValueError("nenhuma configuração da grade cabe no envelope informado")
    X = feature_frame(cands, query, bundle)
    pred_s = np.expm1(bundle["model"].predict(X)) / 1000.0  # alvo do modelo: log1p(ms)
    cands = cands.assign(pred_s=pred_s, pred_lo_s=pred_s / 1.5, pred_hi_s=pred_s * 1.5)
    cands["core_hours"] = cands.total_cores * cands.pred_s / 3600.0
    cands["cost_usd"] = [run_cost_usd(tc * t / 3600.0, tm * t / 3600.0, price)
                         for tc, tm, t in zip(cands.total_cores, cands.total_memory_gb, cands.pred_s)]
    cands["pareto"] = pareto_mask(cands.pred_s.to_numpy(), cands.core_hours.to_numpy())

    # Envelope medido: a célula existe na matriz para o SF nominal mais próximo?
    sf_near = nearest_measured_sf(sf)
    catalog = catalog or load_catalog()
    cells = catalog["cells"]
    measured_cells = set(zip(cells[cells.scale_factor == sf_near].profile, cells[cells.scale_factor == sf_near].workers))
    cands["measured_cell"] = [(p, w) in measured_cells for p, w in zip(cands.profile, cands.workers)]
    if sf < MEASURED_SFS[0] * 0.8 or sf > MEASURED_SFS[-1] * 1.2:
        warnings.append(f"fator de escala {sf:.0f} fora da faixa medida ({MEASURED_SFS[0]}–{MEASURED_SFS[-1]}): extrapolação")
    if query.source in ("explain", "sql"):
        warnings.append("consulta fora do catálogo: só as contagens do plano a descrevem; erro esperado maior (leave-queries-out R² 0,60)")

    key = {"time": "pred_s", "cost": "core_hours", "usd": "cost_usd"}[objective]
    if objective == "time":
        ranked = cands.sort_values("pred_s")
    else:
        # Custo: entre as não dominadas, a mais barata cujo tempo fique a 1,5x da mais rápida.
        limit = cands.pred_s.min() * 1.5
        ranked = cands[(cands.pred_s <= limit)].sort_values(key)
    best = ranked.iloc[0]
    top = ranked.head(top_k).copy()
    top["tie_with_best"] = (top.pred_s <= best.pred_s * (1 + TIE_TOLERANCE))
    if not best.measured_cell:
        warnings.append("a configuração recomendada não foi medida nesta escala: previsão por extrapolação")

    measured = None
    if query.source == "catalogo":
        m = cells[(cells["query"] == query.name) & (cells.scale_factor == sf_near)]
        if not m.empty:
            measured = m.sort_values("measured_s").reset_index(drop=True)
    return Recommendation(cands.sort_values("pred_s").reset_index(drop=True), top, best, warnings, measured)


def explain(query: QuerySpec, row: pd.Series, bundle: Optional[Dict] = None):
    """Valores SHAP (TreeSHAP) da previsão para uma candidata."""
    import shap
    bundle = bundle or load_bundle()
    X = feature_frame(pd.DataFrame([row]), query, bundle)
    explainer = shap.TreeExplainer(bundle["model"])
    return explainer(X)[0]


def regret_if_measured(rec: Recommendation) -> Optional[float]:
    """Se a consulta e a escala estão na matriz: tempo medido da célula
    recomendada sobre o da melhor célula medida, menos um."""
    if rec.measured is None:
        return None
    m = rec.measured
    hit = m[(m.profile == rec.best.profile) & (m.workers == rec.best.workers)]
    if hit.empty:
        return None
    return float(hit.measured_s.iloc[0] / m.measured_s.min() - 1.0)
