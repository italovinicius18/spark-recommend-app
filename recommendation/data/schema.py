"""Canonical data schemas for the recommendation framework."""

from dataclasses import dataclass, field, fields
from typing import Optional

import pandas as pd


@dataclass
class BronzeMetric:
    """Metrics from TPC-DS data generation (dsdgen → Parquet → MinIO)."""

    scale_factor: int
    table_name: str
    parallel_degree: int = 1

    # dsdgen timing
    dsdgen_time_s: float = 0.0

    # CSV (.dat) output
    csv_size_mb: float = 0.0
    csv_row_count: int = 0

    # Parquet conversion
    parquet_conversion_time_s: float = 0.0
    parquet_size_mb: float = 0.0
    compression_ratio: float = 0.0  # csv_size / parquet_size

    # Upload
    upload_time_s: float = 0.0
    upload_throughput_mbps: float = 0.0

    # Total
    total_time_s: float = 0.0

    # Resource usage
    peak_memory_mb: Optional[float] = None
    peak_disk_usage_mb: Optional[float] = None


@dataclass
class SilverMetric:
    """Metrics from Bronze→Silver conversion (Parquet → Delta Lake via Spark)."""

    scale_factor: int
    table_name: str

    # Spark config
    workers: int = 1
    executor_cores: int = 4
    executor_memory_gb: float = 14.0

    # Timing
    read_time_s: float = 0.0
    write_time_s: float = 0.0
    conversion_time_s: float = 0.0  # total

    # Data volumes
    row_count: int = 0
    bronze_size_mb: float = 0.0
    silver_size_mb: float = 0.0
    delta_overhead_ratio: float = 0.0  # silver / bronze
    num_output_files: int = 0

    # Spark metrics (optional)
    shuffle_read_bytes: Optional[float] = None
    shuffle_write_bytes: Optional[float] = None
    gc_time_ms: Optional[float] = None
    executor_run_time_ms: Optional[float] = None
    input_bytes: Optional[float] = None
    output_bytes: Optional[float] = None
    tasks_count: Optional[int] = None
    stages_count: Optional[int] = None

    # MinIO metrics (optional)
    s3_read_throughput_mbps: Optional[float] = None
    s3_write_throughput_mbps: Optional[float] = None
    s3_requests_total: Optional[int] = None

    # Resource metrics (optional)
    driver_cpu_millicores: Optional[float] = None
    driver_memory_mb: Optional[float] = None
    executor_avg_cpu_millicores: Optional[float] = None
    executor_max_memory_mb: Optional[float] = None


@dataclass
class QueryExecution:
    """Metrics from a single TPC-DS query execution (Gold layer)."""

    # Identifiers
    experiment: str  # "exp02", "exp04", "exp05"
    run_id: str
    query: str  # "q01" ... "q99"

    # Configuration (independent variables)
    scale_factor: int
    workers: int
    executor_cores: int
    executor_memory_gb: float

    # Workload provenance ("tpcds" | "tpch") — selects category table + SF sizes
    benchmark: str = "tpcds"

    # Derived configuration
    total_cores: int = 0
    total_memory_gb: float = 0.0

    # Target variable
    duration_ms: float = 0.0
    status: str = "success"  # "success", "timeout", "error"

    # Query classification
    query_category: str = ""  # "A_scan", "B_shuffle", "C_subquery", "D_timeout"

    # Derived features (computed during feature engineering)
    cores_per_gb_data: float = 0.0
    memory_per_gb_data: float = 0.0
    data_per_worker_gb: float = 0.0

    # Cost model (derived)
    cost_core_hours: float = 0.0  # total_cores * (duration_ms / 3_600_000)
    cost_usd: float = 0.0  # cost_core_hours * COST_PER_CORE_HOUR

    # Executor placement (addresses reviewer concern: placement confound)
    # JSON-encoded list of node names where executors were scheduled
    executor_nodes: Optional[str] = None  # e.g. '["ids-es","ids-go","ids-rn"]'
    num_distinct_nodes: Optional[int] = None
    # Fraction of executors on large nodes (48-core) vs small (8-core)
    large_node_fraction: Optional[float] = None
    # Whether any executor is co-located with MinIO (ids-go)
    minio_colocation: Optional[bool] = None

    # Query plan features (from Spark EXPLAIN EXTENDED output)
    # Extracted by run_tpcds_queries_v3.py::extract_plan_features().
    # Used by the model to characterize query complexity structurally
    # (avoiding circular categories — see Threats to Validity).
    # Reference: ZEST (2025) uses plan embeddings; we use discrete features.
    num_joins: Optional[int] = None
    num_broadcast_joins: Optional[int] = None
    num_sort_merge_joins: Optional[int] = None
    num_exchanges: Optional[int] = None  # shuffle stages in physical plan
    num_sorts: Optional[int] = None
    num_aggregates: Optional[int] = None
    num_filters: Optional[int] = None
    num_scans: Optional[int] = None
    num_subqueries: Optional[int] = None
    num_windows: Optional[int] = None
    num_unions: Optional[int] = None
    plan_depth: Optional[int] = None  # max indentation in physical plan
    plan_lines: Optional[int] = None  # total lines in physical plan
    # Derived composite feature: weighted plan complexity score
    plan_complexity: Optional[float] = None

    # AQE baseline flag (addresses reviewer concern: Spark AQE auto-tuner)
    # Per Lyu et al. (VLDB 2024): must compare default+AQE vs recommended+AQE
    aqe_enabled: bool = True

    # Spark stage metrics (optional — available in Exp 04/05 with REST API)
    shuffle_read_bytes: Optional[float] = None
    shuffle_write_bytes: Optional[float] = None
    gc_time_ms: Optional[float] = None
    scheduler_delay_ms: Optional[float] = None
    input_bytes: Optional[float] = None
    output_bytes: Optional[float] = None
    executor_run_time_ms: Optional[float] = None
    executor_cpu_time_ms: Optional[float] = None
    memory_spilled_bytes: Optional[float] = None
    disk_spilled_bytes: Optional[float] = None

    # MinIO metrics (optional)
    minio_read_throughput_mbps: Optional[float] = None
    minio_write_throughput_mbps: Optional[float] = None
    minio_saturation_pct: Optional[float] = None  # throughput / max_throughput

    # Resource metrics (optional)
    driver_cpu_millicores: Optional[float] = None
    driver_memory_mb: Optional[float] = None
    executor_avg_cpu_millicores: Optional[float] = None
    executor_max_memory_mb: Optional[float] = None

    # Temporal control (addresses reviewer concern: temporal confounds)
    run_timestamp: Optional[str] = None  # ISO format
    run_order_in_session: Optional[int] = None  # position within experiment session


def dataclass_to_dataframe(instances: list) -> pd.DataFrame:
    """Convert a list of dataclass instances to a DataFrame."""
    if not instances:
        return pd.DataFrame()
    cols = [f.name for f in fields(instances[0])]
    rows = [{f.name: getattr(inst, f.name) for f in fields(inst)} for inst in instances]
    return pd.DataFrame(rows, columns=cols)
