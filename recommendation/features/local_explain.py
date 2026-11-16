"""Plano físico de uma consulta SQL sem cluster.

O usuário escreve a consulta e informa a magnitude do dado; o modelo precisa
das 14 contagens do plano físico que o Spark produz para essa consulta contra
o esquema e os tamanhos das tabelas. Este módulo sobe uma sessão PySpark
local (um processo, sem executores), cria as tabelas do esquema vazias, com
todas as colunas como `string` (como na Silver real), e grava em cada uma a
estatística de tamanho que ela teria na magnitude pedida. O otimizador então
decide broadcast, sort-merge e filtros de Bloom pelos mesmos limiares do
cluster, e o `EXPLAIN EXTENDED` sai com a mesma estrutura.

Os tamanhos por tabela vêm de `silver_table_sizes.json` (medidos no MinIO em
SF10, SF100 e SF1000) e são interpolados em log-log na magnitude pedida.
Tabelas de dimensão que não crescem com a escala ficam constantes, como no
dsdgen.

Requer `pyspark` instalado (pip extra `web`) e um JDK. A primeira chamada
leva ~20 s para subir a JVM; a sessão fica em cache no processo.
"""
from __future__ import annotations

import ast
import json
import os
import tempfile
from urllib.parse import urlparse
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from recommendation.features.plan_features import extract_plan_features

ROOT = Path(__file__).resolve().parents[2]
SIZES_JSON = Path(__file__).with_name("silver_table_sizes.json")


@lru_cache(maxsize=1)
def tpcds_columns() -> Dict[str, list]:
    """Lê TPCDS_COLUMNS de scripts/bronze_to_silver.py sem importar o módulo
    (ele puxa pyspark e tem efeitos de módulo)."""
    src = (ROOT / "scripts/bronze_to_silver.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "TPCDS_COLUMNS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("TPCDS_COLUMNS não encontrado")


@lru_cache(maxsize=1)
def table_sizes() -> Dict[str, Dict[int, float]]:
    """{tabela: {sf: bytes}} medidos na Silver."""
    raw = json.loads(SIZES_JSON.read_text())
    return {t: {int(k): float(v) for k, v in d.items()} for t, d in raw.items() if not t.startswith("_")}


def size_at(table: str, sf: float) -> float:
    pts = table_sizes().get(table)
    if not pts:
        return 1e6
    xs = np.array(sorted(pts)); ys = np.array([max(pts[k], 1.0) for k in xs])
    return float(np.exp(np.interp(np.log(max(sf, 1e-9)), np.log(xs), np.log(ys))))


@lru_cache(maxsize=1)
def _session():
    from pyspark.sql import SparkSession
    wh = tempfile.mkdtemp(prefix="spark-recommend-wh-")
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    # Mesmos padrões do cluster que importam para a forma do plano: AQE ligado,
    # CBO desligado (padrão do Spark; o controller não o liga), broadcast a
    # 10 MB e Bloom filter a 10 GB (padrões).
    return (SparkSession.builder.master("local[1]").appName("spark-recommend-explain")
            .config("spark.sql.warehouse.dir", wh)
            .config("spark.driver.memory", "1g")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "400")
            .config("spark.sql.adaptive.enabled", "true")
            .getOrCreate())


def _ensure_tables(spark, sf: float, benchmark: str = "tpcds") -> None:
    """Cria as tabelas vazias (uma vez) e dá a cada uma o tamanho da magnitude.

    O tamanho entra pelo próprio sistema de arquivos: cada tabela aponta para
    um diretório com um arquivo esparso do comprimento certo. O Spark estima
    `sizeInBytes` pelo comprimento dos arquivos sem lê-los, e é essa estimativa
    que decide broadcast (10 MB) e filtro de Bloom (10 GB). Arquivo esparso não
    ocupa disco, então 135 GiB de `catalog_sales` custam zero bytes.
    """
    cols = tpcds_columns() if benchmark == "tpcds" else _tpch_columns()
    # A conf devolve um URI (file:/tmp/...); Path() o trataria como caminho relativo.
    base = Path(urlparse(spark.conf.get("spark.sql.warehouse.dir")).path) / benchmark
    for t, cs in cols.items():
        d = base / t
        d.mkdir(parents=True, exist_ok=True)
        f = d / "part-00000.parquet"
        size = max(int(size_at(t, sf)), 1)
        if not f.exists() or f.stat().st_size != size:
            with open(f, "wb") as fh:
                fh.truncate(size)
        if not spark.catalog.tableExists(t):
            names = cs if isinstance(cs[0], str) else [c for c, _ in cs]
            ddl = ", ".join(f"`{c}` string" for c in names)
            spark.sql(f"CREATE TABLE `{t}` ({ddl}) USING parquet LOCATION '{d}'")
        else:
            spark.catalog.refreshTable(t)


def _tpch_columns() -> Dict[str, list]:
    src = (ROOT / "scripts/benchmarks.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "_TPCH_COLUMNS":
            return ast.literal_eval(node.value)
    raise RuntimeError("_TPCH_COLUMNS não encontrado")


def explain_sql(sql: str, sf: float, benchmark: str = "tpcds") -> Dict:
    """EXPLAIN EXTENDED local -> {'plan_text', 'features'}."""
    spark = _session()
    _ensure_tables(spark, sf, benchmark)
    sql = sql.strip().rstrip(";")
    plan_text = spark.sql(f"EXPLAIN EXTENDED {sql}").collect()[0][0]
    return {"plan_text": plan_text, "features": extract_plan_features(plan_text)}


def guess_category(features: Dict[str, float]) -> str:
    """Categoria estrutural pela regra determinística do estudo cruzado
    (recommendation/validation/categories.py), que concorda em ~54% com os
    rótulos manuais do TPC-DS. É um palpite; a interface deixa trocar."""
    from recommendation.validation.categories import derive_category
    return derive_category(features.get("num_subqueries"), features.get("num_exchanges"),
                           features.get("num_sort_merge_joins"))
