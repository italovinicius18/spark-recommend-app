"""Consolidate all experiment data into a unified training dataset."""

import logging
from pathlib import Path

import pandas as pd

from recommendation.config import CONSOLIDATED_DIR
from recommendation.data.loaders import load_all_query_executions
from recommendation.data.schema import dataclass_to_dataframe

logger = logging.getLogger(__name__)


def consolidate(
    output_dir: Path | None = None,
    include_contaminated: bool = False,
) -> pd.DataFrame:
    """Load the valid experiments, deduplicate, and save as parquet.

    Este é o lugar único dos dados da dissertação. Por padrão entra só o que
    rodou sobre a Silver reparada (exp05, campanhas de re-run); exp02 e exp04
    ficam de fora porque são inválidos em 36 das 99 queries e não podem ser
    re-executados. Ver `load_all_query_executions`.

    Deduplication: if the same (scale_factor, workers, executor_cores,
    executor_memory_gb, query, run_id) appears in multiple experiments,
    prefer the later experiment (exp05 > exp04 > exp02).
    """
    if output_dir is None:
        output_dir = CONSOLIDATED_DIR

    records = load_all_query_executions(include_contaminated=include_contaminated)
    if not records:
        # Silêncio aqui já custou caro: `load_exp05` apontava para um diretório
        # vazio e a consolidação seguia adiante sem dados de exp05. Falhar alto.
        raise RuntimeError(
            "Nenhum registro carregado. Verifique CLEAN_METRICS_DIRS — os "
            "relatórios das campanhas são .json.gz e ficam em metrics/rerun-*."
        )

    df = dataclass_to_dataframe(records)

    # Deduplicate: keep the latest experiment's data when overlapping
    experiment_priority = {"exp02": 0, "exp04": 1, "exp05": 2}
    df["_priority"] = df["experiment"].map(experiment_priority).fillna(0)

    # `run_order_in_session` é obrigatório na chave: dentro de um mesmo run a
    # query aparece uma vez por try (TRIES=3), todas com o mesmo run_id. Sem
    # ele o dedup descarta 2 de cada 3 execuções — some justamente a
    # repetição que sustenta variância e intervalo de confiança.
    dedup_cols = [
        "scale_factor",
        "workers",
        "executor_cores",
        "executor_memory_gb",
        "query",
        "run_id",
        "run_order_in_session",
    ]
    df = df.sort_values("_priority", ascending=False)
    before = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first")
    df = df.drop(columns=["_priority"])
    after = len(df)

    if before != after:
        logger.info("Deduplicated: %d → %d records (%d removed)", before, after, before - after)

    # Uma célula (SF × forma de executor × workers) deve aparecer uma vez só.
    # Se aparecer duas, alguma re-execução entrou junto com a original e as
    # duas vão para dentro das médias.
    cells = df.groupby(
        ["experiment", "scale_factor", "executor_cores", "executor_memory_gb", "workers"]
    )["run_id"].nunique()
    repeated = cells[cells > 1]
    if not repeated.empty:
        logger.warning("Células com mais de um run_id:\n%s", repeated.to_string())

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "training_data.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved %d records to %s", len(df), parquet_path)

    _write_manifest(df, output_dir / "manifest.csv")

    # Print summary
    _print_summary(df)

    return df


def load_consolidated(output_dir: Path | None = None) -> pd.DataFrame:
    """Load previously consolidated data from parquet."""
    if output_dir is None:
        output_dir = CONSOLIDATED_DIR

    parquet_path = output_dir / "training_data.parquet"
    if not parquet_path.exists():
        logger.warning("No consolidated data found at %s. Run consolidate() first.", parquet_path)
        return pd.DataFrame()

    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d records from %s", len(df), parquet_path)
    return df


def _write_manifest(df: pd.DataFrame, path: Path) -> None:
    """Escreve uma linha por célula, para auditar a proveniência do dataset.

    O parquet sozinho não diz de qual run cada célula veio nem quantas
    execuções ela trouxe. O manifesto diz, e é o que permite conferir a
    cobertura da matriz sem reabrir os relatórios.
    """
    manifest = (
        df.groupby(
            [
                "experiment",
                "scale_factor",
                "executor_cores",
                "executor_memory_gb",
                "workers",
                "run_id",
            ]
        )
        .agg(
            executions=("query", "size"),
            queries=("query", "nunique"),
            success=("status", lambda s: (s == "success").sum()),
            timeout=("status", lambda s: (s == "timeout").sum()),
            failed=("status", lambda s: (s == "failed").sum()),
            wall_hours=("duration_ms", lambda s: round(s.sum() / 3_600_000, 2)),
            run_timestamp=("run_timestamp", "first"),
        )
        .reset_index()
        .sort_values(["scale_factor", "executor_cores", "workers"])
    )
    manifest.to_csv(path, index=False)
    logger.info("Manifesto com %d células em %s", len(manifest), path)


def _print_summary(df: pd.DataFrame) -> None:
    """Print a summary of the consolidated dataset."""
    print(f"\n{'=' * 60}")
    print("Consolidated Dataset Summary")
    print(f"{'=' * 60}")
    print(f"Total records: {len(df):,}")
    print(f"Experiments: {sorted(df['experiment'].unique())}")
    print(f"Scale factors: {sorted(df['scale_factor'].unique())}")
    print(f"Executor configs: {sorted(df['executor_cores'].unique())} cores")
    print(f"Worker counts: {sorted(df['workers'].unique())}")
    print(f"Queries: {df['query'].nunique()}")
    print(f"Status distribution:")
    for status, count in df["status"].value_counts().items():
        print(f"  {status}: {count:,}")

    print(f"\nRecords per experiment:")
    for exp, count in df["experiment"].value_counts().sort_index().items():
        print(f"  {exp}: {count:,}")

    print(f"\nRecords per scale factor:")
    for sf, count in df.groupby("scale_factor").size().items():
        print(f"  SF{sf}: {count:,}")

    print(f"{'=' * 60}\n")
