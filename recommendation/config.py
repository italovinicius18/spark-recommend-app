"""Framework configuration: paths, constants, and query categories."""

from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXP02_DIR = EXPERIMENTS_DIR / "02-k8s-rnp-weak-scaling"
EXP04_DIR = EXPERIMENTS_DIR / "04-semish-2026-vertical-scaling"
EXP05_DIR = EXPERIMENTS_DIR / "05-dissertation-full-matrix"
EXP06_DIR = EXPERIMENTS_DIR / "06-tpch-cross-workload"

CONSOLIDATED_DIR = PROJECT_ROOT / "data" / "consolidated"
MODELS_DIR = PROJECT_ROOT / "models"

# ── Dados válidos para a dissertação ─────────────────────────────────────────
#
# A Silver `web_sales` tinha colunas desalinhadas em todos os SFs até ser
# regravada em 2026-08-19 (bug de ordem em TPCDS_COLUMNS, corrigido em
# 6c43241). 36 das 99 queries TPC-DS tocam `web_sales`, então tudo que rodou
# antes da regravação é semanticamente inválido em ~36% das medições. Isso
# inclui exp02 e exp04 inteiros, que não têm como ser re-executados.
#
# Só as campanhas abaixo rodaram sobre dados reparados. Elas são a única fonte
# do dataset da dissertação; ver `load_exp05`.
CLEAN_METRICS_DIRS = [
    EXP05_DIR / "metrics" / "rerun-202608",
    EXP05_DIR / "metrics" / "rerun-202609",
    # xlarge em SF750 + linha inteira de SF1000 (12 a 21/set/2026). Traz o
    # relatório danificado da sf1000-xlarge-w4 de propósito, como evidência;
    # `load_exp05` o rejeita sozinho (229 não-success < 5 s) e fica com o
    # re-run de 21/set.
    EXP05_DIR / "metrics" / "xlarge-bigsf-202609",
]

# Portões de validação de célula (ver `load_exp05`).
CELL_EXPECTED_EXECUTIONS = 297  # 99 queries × 3 tries
# Não-success com duração quase zero é assinatura de app abortado: o run morre
# e todas as queries seguintes falham em cascata. Isso é dano de infra, não
# resultado — o wall da célula mede trabalho que nunca foi feito.
DAMAGE_FASTFAIL_MS = 5_000
DAMAGE_FASTFAIL_COUNT = 5

# ── TPC-DS Scale Factor sizes (approximate, in GB of raw Parquet) ────────────

SF_SIZE_GB = {
    1: 0.3,
    10: 3.2,
    50: 13.0,
    100: 27.0,
    250: 135.0,
    500: 192.0,
    750: 300.0,
    1000: 400.0,
}

# TPC-H Parquet (snappy) sizes in GiB, MEASURED on MinIO after each Exp 06
# bronze run (mc du, 2026-07-24): sf100=38, sf250=94, sf500=188, sf750=283.
# Ratio is stable at ~0.377 GiB per SF unit; SF1 / 10 / 50 / 1000 are linear
# extrapolations from that ratio (SF1000 pending its own generation).
SF_SIZE_GB_TPCH = {
    1: 0.38, 10: 3.8, 50: 18.9, 100: 38.0,
    250: 94.0, 500: 188.0, 750: 283.0, 1000: 377.0,
}

_SF_SIZE_BY_BENCHMARK = {"tpcds": SF_SIZE_GB, "tpch": SF_SIZE_GB_TPCH}


def sf_size_gb(sf: int, benchmark: str = "tpcds") -> float:
    """Raw Parquet GB for an SF; unknown SFs fall back to float(sf)."""
    return _SF_SIZE_BY_BENCHMARK[benchmark].get(sf, float(sf))


# ── Executor configurations ─────────────────────────────────────────────────

EXECUTOR_CONFIGS = {
    "xsmall": {"cores": 2, "memory_gb": 7},
    "small": {"cores": 4, "memory_gb": 14},
    "medium": {"cores": 8, "memory_gb": 28},
    "large": {"cores": 12, "memory_gb": 48},
    "xlarge": {"cores": 24, "memory_gb": 96},
}

# ── Cost model ───────────────────────────────────────────────────────────────
# Normalized cost per core-hour (unitless, for relative comparison).
# Based on AWS EC2 on-demand pricing as reference (~$0.05/core-hour for m5.xlarge).
# For academic clusters, this represents opportunity cost (cluster time).

COST_PER_CORE_HOUR = 0.05  # USD equivalent, adjustable

# ── Cluster topology (RNP) ──────────────────────────────────────────────────

CLUSTER_NODES = {
    "ids-es": {"cores": 48, "memory_gi": 187, "type": "large"},
    "ids-go": {"cores": 48, "memory_gi": 187, "type": "large", "has_minio": True},
    "ids-rn": {"cores": 48, "memory_gi": 187, "type": "large"},
    "ids-rj": {"cores": 24, "memory_gi": 176, "type": "medium"},
    "ids-pb": {"cores": 24, "memory_gi": 172, "type": "medium"},
    "whx-ba": {"cores": 8, "memory_gi": 15, "type": "small"},
    "whx-mg": {"cores": 8, "memory_gi": 15, "type": "small"},
    "whx-rn": {"cores": 8, "memory_gi": 15, "type": "small"},
}

# ── AQE (Adaptive Query Execution) ──────────────────────────────────────────
# Spark AQE is an internal auto-tuner. We test with AQE ON (default) and
# document it as a baseline that our framework complements (not replaces).

AQE_ENABLED_BY_DEFAULT = True

# ── Diminishing returns threshold ────────────────────────────────────────────
# A config is considered to have diminishing returns if adding resources
# increases cost by >COST_INCREASE_PCT but improves time by <TIME_IMPROVE_PCT.

DIMINISHING_RETURNS_COST_INCREASE_PCT = 10.0
DIMINISHING_RETURNS_TIME_IMPROVE_PCT = 5.0

# ── Query categories (from TPC-DS query *structure*, not observed behavior) ──
# Classified by SQL structure: num_joins, subqueries, exchanges from EXPLAIN.
# This avoids the "circular categories" threat (categories derived from the
# same data used to train the model).  Categories will be validated/refined
# once query plan features are available from Exp 05 EXPLAIN output.
#
# Literature reference: ZEST (2025) embeds logical plans to capture query
# similarity.  We use discrete structural categories as a lighter-weight
# alternative, with plan features (num_joins, etc.) as continuous features.

QUERY_CATEGORIES = {
    "A_scan": [
        "q1", "q3", "q6", "q7", "q10", "q12", "q13", "q15", "q19", "q20",
        "q22", "q26", "q28", "q29", "q33", "q37", "q40", "q42", "q43", "q46",
        "q52", "q53", "q55", "q58", "q59", "q61", "q65", "q68", "q71", "q73",
        "q79", "q82", "q84", "q85", "q96", "q97", "q98", "q99",
    ],
    # q95 added here: web_sales self-join (8 joins / 7-10 exchanges, ~0 subqueries) is
    # structurally shuffle-heavy. D_timeout was a behavioral label from early pipeline
    # versions (q95 timed out frequently at high SFs); removed per banca §2.5 requirement
    # that categories reflect SQL structure, not observed execution behavior.
    "B_shuffle": [
        "q11", "q16", "q17", "q18", "q25", "q27", "q34", "q44", "q47", "q48",
        "q56", "q57", "q60", "q63", "q64", "q66", "q74", "q75", "q76", "q80",
        "q81", "q86", "q87", "q88", "q89", "q90", "q91", "q92", "q93", "q94",
        "q95",
    ],
    "C_subquery": [
        "q2", "q4", "q5", "q8", "q9", "q14a", "q14b", "q21", "q23a", "q23b",
        "q30", "q31", "q32", "q35", "q36", "q38", "q39a", "q39b", "q41", "q45",
        "q49", "q50", "q51", "q54", "q62", "q67", "q69", "q70", "q72", "q77",
        "q78", "q83",
    ],
}

# TPC-H query categories by SQL structure (same A/B/C scheme as TPC-DS).
# PROVISIONAL hand-assignment — regenerate from EXPLAIN plan features once
# TPC-H plans are collected (paper methodology requirement).
QUERY_CATEGORIES_TPCH = {
    "A_scan": ["q1", "q6", "q12", "q14", "q15", "q19"],
    "B_shuffle": ["q3", "q5", "q7", "q8", "q9", "q10", "q21"],
    "C_subquery": ["q2", "q4", "q11", "q13", "q16", "q17", "q18", "q20", "q22"],
}

_QUERY_TO_CATEGORY_TPCH = {}
for _cat, _queries in QUERY_CATEGORIES_TPCH.items():
    for _q in _queries:
        _QUERY_TO_CATEGORY_TPCH[_q] = _cat

# ── Model baselines ─────────────────────────────────────────────────────────
# Literature (CherryPick, LOCAT, Rover, Li et al.) uses Bayesian Optimization.
# We include BO as a baseline to justify the supervised (XGBoost) choice given
# our large dataset (60K+ points).  BO excels with <100 observations; XGBoost
# leverages the full dataset.

MODEL_BASELINES = [
    "default_config",       # Spark defaults + AQE
    "random_config",        # Random valid config from the search space
    "category_median",      # Median duration per query category
    "bayesian_optimization", # GP + EI (CherryPick-style) — key literature baseline
]

# ── AQE experiment flags ────────────────────────────────────────────────────
# Per Lyu et al. (VLDB 2024) and AQORA (2025), AQE is itself an auto-tuner.
# We must show our framework adds value *on top* of AQE, not instead of it.
# Subset of runs with AQE disabled serves as control group.

AQE_CONTROL_CONFIGS = [
    # (sf, config, workers) — small subset to show AQE vs no-AQE delta
    (100, "small", 4),
    (100, "medium", 4),
    (500, "small", 4),
]

# Reverse lookup: query_id → category
_QUERY_TO_CATEGORY = {}
for _cat, _queries in QUERY_CATEGORIES.items():
    for _q in _queries:
        _QUERY_TO_CATEGORY[_q] = _cat


def query_category(query_id: str, benchmark: str = "tpcds") -> str:
    """Return the category for a query ID (e.g., 'q01' -> 'A_scan')."""
    normalized = query_id.lstrip("q").lstrip("0")
    normalized = f"q{normalized}" if normalized else query_id
    table = _QUERY_TO_CATEGORY if benchmark == "tpcds" else _QUERY_TO_CATEGORY_TPCH
    return table.get(normalized, table.get(query_id, "B_shuffle"))
