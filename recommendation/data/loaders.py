"""Data loaders for each experiment format."""

import csv
import gzip
import json
import logging
from pathlib import Path
from typing import Optional

from recommendation.config import (
    CELL_EXPECTED_EXECUTIONS,
    CLEAN_METRICS_DIRS,
    DAMAGE_FASTFAIL_COUNT,
    DAMAGE_FASTFAIL_MS,
    EXP02_DIR,
    EXP04_DIR,
    EXP05_DIR,
    EXP06_DIR,
    query_category,
    sf_size_gb,
)
from recommendation.data.schema import QueryExecution

logger = logging.getLogger(__name__)


def _read_benchmark_report(path: Path) -> Optional[dict]:
    """Read a benchmark report, whichever of the three layouts it uses.

    Three coexist no repositório:
      - arquivo único `benchmark-<id>.json` (PUT via boto3, formato atual);
      - arquivo `benchmark-<id>.json.gz` (arquivado por
        `archive_campaign_results.sh`, é o que está versionado);
      - diretório `benchmark-<id>.json/part-00000` (saveAsTextFile antigo).

    O terceiro é armadilhado: `mc cat` no nome devolve vazio em silêncio, e foi
    por isso que 61 relatórios ficaram meses sem ser arquivados.
    """
    try:
        if path.is_dir():
            parts = sorted(path.glob("part-*"))
            if not parts:
                logger.warning("Empty Spark output dir: %s", path)
                return None
            return json.loads(parts[0].read_text())
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as fh:
                return json.load(fh)
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def _parse_memory_gb(mem_str: str) -> float:
    """Parse memory string like '14g' or '7g' to float GB."""
    mem_str = str(mem_str).strip().lower()
    if mem_str.endswith("g"):
        return float(mem_str[:-1])
    if mem_str.endswith("m"):
        return float(mem_str[:-1]) / 1024
    return float(mem_str)


def _compute_derived(qe: QueryExecution) -> QueryExecution:
    """Fill in derived fields on a QueryExecution."""
    qe.total_cores = qe.workers * qe.executor_cores
    qe.total_memory_gb = qe.workers * qe.executor_memory_gb
    qe.query_category = query_category(qe.query, benchmark=qe.benchmark)

    sf_gb = sf_size_gb(qe.scale_factor, qe.benchmark)
    if sf_gb > 0:
        qe.cores_per_gb_data = qe.total_cores / sf_gb
        qe.memory_per_gb_data = qe.total_memory_gb / sf_gb
    if qe.workers > 0:
        qe.data_per_worker_gb = sf_gb / qe.workers

    return qe


# ── Experiment 02: Weak Scaling (CSV) ────────────────────────────────────────


def load_exp02(
    results_dir: Optional[Path] = None,
) -> list[QueryExecution]:
    """Load Experiment 02 data from all_query_times.csv.

    Exp 02 uses fixed executor config: 4 cores, 14g memory.
    """
    if results_dir is None:
        results_dir = EXP02_DIR / "results"

    csv_path = results_dir / "all_query_times.csv"
    if not csv_path.exists():
        logger.warning("Exp 02 CSV not found: %s", csv_path)
        return []

    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qe = QueryExecution(
                experiment="exp02",
                run_id=row["run_timestamp"],
                query=row["query"],
                scale_factor=int(row["scale_factor"]),
                workers=int(row["workers"]),
                executor_cores=4,
                executor_memory_gb=14.0,
                duration_ms=float(row["time_ms"]),
                status=row.get("status", "success"),
            )
            records.append(_compute_derived(qe))

    logger.info("Loaded %d records from Exp 02", len(records))
    return records


# ── Experiment 04: Vertical Scaling (JSON v3) ────────────────────────────────


def load_exp04(
    metrics_dir: Optional[Path] = None,
) -> list[QueryExecution]:
    """Load Experiment 04 data from per-SF JSON files.

    Each JSON contains configuration, spark_config, and detailed_results.
    """
    if metrics_dir is None:
        metrics_dir = EXP04_DIR / "metrics"

    if not metrics_dir.exists():
        logger.warning("Exp 04 metrics dir not found: %s", metrics_dir)
        return []

    records = []

    for json_path in sorted(metrics_dir.rglob("benchmark-*.json")):
        # Two storage layouts coexist:
        #   - New (single-file boto3 PUT): benchmark-<id>.json is a file.
        #   - Old (Spark saveAsTextFile): benchmark-<id>.json is a directory
        #     containing part-00000 with the JSON body.
        try:
            if json_path.is_dir():
                parts = sorted(json_path.glob("part-*"))
                if not parts:
                    logger.warning("Empty Spark output dir: %s", json_path)
                    continue
                data = json.loads(parts[0].read_text())
            else:
                data = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", json_path, e)
            continue

        config = data.get("configuration", {})
        spark_config = data.get("spark_config", {})

        sf = int(config.get("scale_factor", 0))
        workers = int(config.get("workers", spark_config.get("executor_instances", 0)))
        executor_cores = int(spark_config.get("executor_cores", config.get("executor_cores", 4)))
        executor_memory_gb = _parse_memory_gb(
            spark_config.get("executor_memory", config.get("executor_memory", "14g"))
        )

        run_id = json_path.stem.replace("benchmark-", "")

        detailed = data.get("detailed_results", [])
        for result in detailed:
            query = result.get("query", "")
            status = result.get("status", "success")
            duration_ms = float(result.get("execution_time_ms", 0))

            if status == "timeout":
                duration_ms = float(result.get("execution_time_ms", 1800000))

            qe = QueryExecution(
                experiment="exp04",
                run_id=run_id,
                query=query,
                scale_factor=sf,
                workers=workers,
                executor_cores=executor_cores,
                executor_memory_gb=executor_memory_gb,
                duration_ms=duration_ms,
                status=status,
            )
            records.append(_compute_derived(qe))

    logger.info("Loaded %d records from Exp 04", len(records))
    return records


# ── Experiments 05/06: v3 JSON metrics (same format as Exp 04) ───────────────


def _cell_rejection(data: dict) -> Optional[str]:
    """Return why this cell must not enter the dataset, or None if it is valid.

    Duas coisas diferentes se parecem no relatório e precisam ser separadas:

      - **Dano de infra**: o app Spark foi abortado no meio e todas as queries
        seguintes falharam em cascata, cada uma em ~0 s. A célula não mediu o
        que diz medir e o wall dela é menor porque houve menos trabalho, não
        porque foi mais rápida. Fora.
      - **Falha genuína**: a query rodou, gastou tempo real e falhou por
        limitação da configuração. Isso *é* resultado e fica no dataset.
    """
    summary = data.get("summary", {})
    detailed = data.get("detailed_results", [])

    total = int(summary.get("total_executions", len(detailed)))
    if total != CELL_EXPECTED_EXECUTIONS:
        return f"{total} execuções (esperado {CELL_EXPECTED_EXECUTIONS})"

    fastfail = sum(
        1
        for r in detailed
        if r.get("status") != "success"
        and float(r.get("execution_time_ms", 0)) < DAMAGE_FASTFAIL_MS
    )
    if fastfail >= DAMAGE_FASTFAIL_COUNT:
        return f"{fastfail} falhas em <{DAMAGE_FASTFAIL_MS} ms (app abortado)"

    return None


def _load_v3_metrics(
    search_dirs: list[Path],
    experiment: str,
    benchmark: str = "tpcds",
    validate_cells: bool = True,
) -> list[QueryExecution]:
    """Load v3-format benchmark JSONs (layout shared by Exp 04/05/06).

    `validate_cells` aplica os portões de `_cell_rejection` (297 execuções,
    sem cascata de falhas < 5 s). Vale para o TPC-DS do Exp 05; o TPC-H do
    Exp 06 tem 22 queries × 3 tries = 66 execuções por célula e mantém as
    células inviáveis no corpus de propósito (a falha é resultado do paper),
    então passa `validate_cells=False`.
    """
    paths: list[Path] = []
    for d in search_dirs:
        if not d.exists():
            logger.warning("Exp 05 metrics dir not found: %s", d)
            continue
        for pattern in ("benchmark-*.json", "benchmark-*.json.gz"):
            for p in d.rglob(pattern):
                # Diretórios cujo nome começa com "_" guardam material
                # auxiliar (`_contaminated/`, `_timeouts_resolved/`,
                # `_aggregated/`) e não são runs adicionais.
                if any(part.startswith("_") for part in p.relative_to(d).parts[:-1]):
                    continue
                paths.append(p)

    records = []
    accepted = rejected = 0

    for json_path in sorted(paths):
        data = _read_benchmark_report(json_path)
        if data is None:
            rejected += 1
            continue

        reason = _cell_rejection(data) if validate_cells else None
        if reason is not None:
            logger.warning("Célula descartada (%s): %s", reason, json_path.name)
            rejected += 1
            continue
        accepted += 1

        config = data.get("configuration", {})
        spark_config = data.get("spark_config", {})

        sf = int(config.get("scale_factor", 0))
        workers = int(config.get("workers", spark_config.get("executor_instances", 0)))
        executor_cores = int(spark_config.get("executor_cores", config.get("executor_cores", 4)))
        executor_memory_gb = _parse_memory_gb(
            spark_config.get("executor_memory", config.get("executor_memory", "14g"))
        )

        # `.stem` de "benchmark-X.json.gz" ainda termina em ".json".
        run_id = json_path.name.split(".")[0].replace("benchmark-", "")

        timestamp = data.get("timestamp", "")
        detailed = data.get("detailed_results", [])
        for order, result in enumerate(detailed):
            query = result.get("query", "")
            status = result.get("status", "success")
            duration_ms = float(result.get("execution_time_ms", 0))

            def _opt_int(key: str) -> Optional[int]:
                v = result.get(key)
                return int(v) if v is not None else None

            def _opt_float(key: str) -> Optional[float]:
                v = result.get(key)
                return float(v) if v is not None else None

            num_joins = _opt_int("num_joins")
            num_broadcasts = _opt_int("num_broadcast_joins")
            num_smj = _opt_int("num_sort_merge_joins")
            num_exchanges = _opt_int("num_exchanges")
            num_sorts = _opt_int("num_sorts")
            num_aggs = _opt_int("num_aggregates")
            num_filters = _opt_int("num_filters")
            num_scans = _opt_int("num_scans")
            num_subqueries = _opt_int("num_subqueries")
            num_windows = _opt_int("num_windows")
            num_unions = _opt_int("num_unions")
            plan_depth = _opt_int("plan_depth")
            plan_lines = _opt_int("plan_lines")

            plan_complexity: Optional[float] = None
            if num_joins is not None:
                plan_complexity = (
                    (num_joins or 0) * 3
                    + (num_exchanges or 0) * 2
                    + (num_aggs or 0) * 1.5
                    + (num_subqueries or 0) * 4
                    + (num_sorts or 0)
                )

            qe = QueryExecution(
                experiment=experiment,
                benchmark=benchmark,
                run_id=run_id,
                query=query,
                scale_factor=sf,
                workers=workers,
                executor_cores=executor_cores,
                executor_memory_gb=executor_memory_gb,
                duration_ms=duration_ms,
                status=status,
                num_joins=num_joins,
                num_broadcast_joins=num_broadcasts,
                num_sort_merge_joins=num_smj,
                num_exchanges=num_exchanges,
                num_sorts=num_sorts,
                num_aggregates=num_aggs,
                num_filters=num_filters,
                num_scans=num_scans,
                num_subqueries=num_subqueries,
                num_windows=num_windows,
                num_unions=num_unions,
                plan_depth=plan_depth,
                plan_lines=plan_lines,
                plan_complexity=plan_complexity,
                shuffle_read_bytes=_opt_float("shuffle_read_bytes"),
                shuffle_write_bytes=_opt_float("shuffle_write_bytes"),
                gc_time_ms=_opt_float("gc_time_ms"),
                input_bytes=_opt_float("input_bytes"),
                run_timestamp=timestamp,
                run_order_in_session=order,
            )
            records.append(_compute_derived(qe))

    logger.info(
        "%s: %d registros de %d células (%d células descartadas)",
        experiment,
        len(records),
        accepted,
        rejected,
    )
    return records


def load_exp05(
    metrics_dir: Optional[Path] = None,
    metrics_dirs: Optional[list[Path]] = None,
) -> list[QueryExecution]:
    """Load Experiment 05 data — apenas células sobre a Silver reparada.

    O default são os diretórios das campanhas de re-run
    (`CLEAN_METRICS_DIRS`), **não** `metrics/gold/`. O `gold/` guarda os
    relatórios de abr-jun/2026, que rodaram sobre a `web_sales` desalinhada e
    são inválidos em 36 das 99 queries; apontar para lá reintroduz a
    contaminação sem nenhum erro visível.
    """
    if metrics_dir is not None:
        search_dirs = [metrics_dir]
    elif metrics_dirs is not None:
        search_dirs = list(metrics_dirs)
    else:
        search_dirs = list(CLEAN_METRICS_DIRS)
    return _load_v3_metrics(search_dirs, "exp05")


def load_tpch(metrics_dir: Optional[Path] = None) -> list[QueryExecution]:
    """Load Experiment 06 (TPC-H cross-workload validation) data."""
    if metrics_dir is None:
        metrics_dir = EXP06_DIR / "metrics" / "gold"
    return _load_v3_metrics(
        [metrics_dir], "exp06-tpch", benchmark="tpch", validate_cells=False
    )


# ── Load all experiments ─────────────────────────────────────────────────────


def load_all_query_executions(
    include_contaminated: bool = False,
) -> list[QueryExecution]:
    """Load query execution data. Por padrão, só o que rodou sobre dados bons.

    exp02 e exp04 rodaram sobre a `web_sales` desalinhada e não têm como ser
    re-executados — os resultados deles já saíram em artigos publicados. Ficam
    de fora do dataset da dissertação. `include_contaminated=True` os traz de
    volta para inspeção do histórico, nunca para treinar modelo ou gerar
    figura.
    """
    all_records = []
    if include_contaminated:
        logger.warning(
            "Incluindo exp02/exp04: inválidos em 36 das 99 queries (web_sales "
            "desalinhada). Não use para treino nem para figura."
        )
        all_records.extend(load_exp02())
        all_records.extend(load_exp04())
    all_records.extend(load_exp05())
    logger.info("Total records loaded: %d", len(all_records))
    return all_records
