#!/usr/bin/env python3
"""
Bronze to Silver Layer Conversion for Kubernetes environment.
Converts TPC-DS data from Bronze (Parquet) to Silver layer using Delta Lake.

Adapted from lhbench-v2 for Kubernetes deployment.
"""

import sys
import os
import json
import time
import platform
from datetime import datetime
from pyspark.sql import SparkSession

try:
    import boto3
    from botocore.client import Config as BotoConfig
    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import current as current_benchmark
_BENCHMARK = current_benchmark()


def cast_select_exprs(col_types: dict) -> list:
    """Build Spark SQL CAST expressions from a {col: type} map."""
    return [f"CAST({c} AS {t}) AS {c}" for c, t in col_types.items()]


# Configuration from arguments or environment
SCALE_FACTOR = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LAKEHOUSE_TYPE = sys.argv[2] if len(sys.argv) > 2 else "delta"
FILE_FORMAT = sys.argv[3] if len(sys.argv) > 3 else "parquet"

# MinIO Configuration for Kubernetes
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

METRICS_PATH = os.getenv("METRICS_PATH", f"/opt/spark/metrics/sf{SCALE_FACTOR}")
METRICS_BUCKET = os.getenv("METRICS_BUCKET", "spark-logs")
METRICS_PREFIX = os.getenv("METRICS_PREFIX", "exp05/silver")


def _s3_client():
    if not _HAS_BOTO3:
        return None
    endpoint = MINIO_ENDPOINT if MINIO_ENDPOINT.startswith("http") else f"http://{MINIO_ENDPOINT}"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",
    )


def _s3_prefix_size(s3, bucket: str, prefix: str) -> tuple:
    """Return (total_bytes, object_count) under s3://bucket/prefix."""
    if s3 is None:
        return 0, 0
    total = 0
    count = 0
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                total += int(obj.get("Size", 0))
                count += 1
    except Exception as e:
        print(f"  WARN: could not list s3://{bucket}/{prefix}: {e}")
    return total, count

# TPC-DS column names per table (used to fix Bronze data generated without column names)
TPCDS_COLUMNS = {
    "call_center": ["cc_call_center_sk","cc_call_center_id","cc_rec_start_date","cc_rec_end_date","cc_closed_date_sk","cc_open_date_sk","cc_name","cc_class","cc_employees","cc_sq_ft","cc_hours","cc_manager","cc_mkt_id","cc_mkt_class","cc_mkt_desc","cc_market_manager","cc_division","cc_division_name","cc_company","cc_company_name","cc_street_number","cc_street_name","cc_street_type","cc_suite_number","cc_city","cc_county","cc_state","cc_zip","cc_country","cc_gmt_offset","cc_tax_percentage"],
    "catalog_page": ["cp_catalog_page_sk","cp_catalog_page_id","cp_start_date_sk","cp_end_date_sk","cp_department","cp_catalog_number","cp_catalog_page_number","cp_description","cp_type"],
    "catalog_returns": ["cr_returned_date_sk","cr_returned_time_sk","cr_item_sk","cr_refunded_customer_sk","cr_refunded_cdemo_sk","cr_refunded_hdemo_sk","cr_refunded_addr_sk","cr_returning_customer_sk","cr_returning_cdemo_sk","cr_returning_hdemo_sk","cr_returning_addr_sk","cr_call_center_sk","cr_catalog_page_sk","cr_ship_mode_sk","cr_warehouse_sk","cr_reason_sk","cr_order_number","cr_return_quantity","cr_return_amount","cr_return_tax","cr_return_amt_inc_tax","cr_fee","cr_return_ship_cost","cr_refunded_cash","cr_reversed_charge","cr_store_credit","cr_net_loss"],
    "catalog_sales": ["cs_sold_date_sk","cs_sold_time_sk","cs_ship_date_sk","cs_bill_customer_sk","cs_bill_cdemo_sk","cs_bill_hdemo_sk","cs_bill_addr_sk","cs_ship_customer_sk","cs_ship_cdemo_sk","cs_ship_hdemo_sk","cs_ship_addr_sk","cs_call_center_sk","cs_catalog_page_sk","cs_ship_mode_sk","cs_warehouse_sk","cs_item_sk","cs_promo_sk","cs_order_number","cs_quantity","cs_wholesale_cost","cs_list_price","cs_sales_price","cs_ext_discount_amt","cs_ext_sales_price","cs_ext_wholesale_cost","cs_ext_list_price","cs_ext_tax","cs_coupon_amt","cs_ext_ship_cost","cs_net_paid","cs_net_paid_inc_tax","cs_net_paid_inc_ship","cs_net_paid_inc_ship_tax","cs_net_profit"],
    "customer": ["c_customer_sk","c_customer_id","c_current_cdemo_sk","c_current_hdemo_sk","c_current_addr_sk","c_first_shipto_date_sk","c_first_sales_date_sk","c_salutation","c_first_name","c_last_name","c_preferred_cust_flag","c_birth_day","c_birth_month","c_birth_year","c_birth_country","c_login","c_email_address","c_last_review_date_sk"],
    "customer_address": ["ca_address_sk","ca_address_id","ca_street_number","ca_street_name","ca_street_type","ca_suite_number","ca_city","ca_county","ca_state","ca_zip","ca_country","ca_gmt_offset","ca_location_type"],
    "customer_demographics": ["cd_demo_sk","cd_gender","cd_marital_status","cd_education_status","cd_purchase_estimate","cd_credit_rating","cd_dep_count","cd_dep_employed_count","cd_dep_college_count"],
    "date_dim": ["d_date_sk","d_date_id","d_date","d_month_seq","d_week_seq","d_quarter_seq","d_year","d_dow","d_moy","d_dom","d_qoy","d_fy_year","d_fy_quarter_seq","d_fy_week_seq","d_day_name","d_quarter_name","d_holiday","d_weekend","d_following_holiday","d_first_dom","d_last_dom","d_same_day_ly","d_same_day_lq","d_current_day","d_current_week","d_current_month","d_current_quarter","d_current_year"],
    "household_demographics": ["hd_demo_sk","hd_income_band_sk","hd_buy_potential","hd_dep_count","hd_vehicle_count"],
    "income_band": ["ib_income_band_sk","ib_lower_bound","ib_upper_bound"],
    "inventory": ["inv_date_sk","inv_item_sk","inv_warehouse_sk","inv_quantity_on_hand"],
    "item": ["i_item_sk","i_item_id","i_rec_start_date","i_rec_end_date","i_item_desc","i_current_price","i_wholesale_cost","i_brand_id","i_brand","i_class_id","i_class","i_category_id","i_category","i_manufact_id","i_manufact","i_size","i_formulation","i_color","i_units","i_container","i_manager_id","i_product_name"],
    "promotion": ["p_promo_sk","p_promo_id","p_start_date_sk","p_end_date_sk","p_item_sk","p_cost","p_response_target","p_promo_name","p_channel_dmail","p_channel_email","p_channel_catalog","p_channel_tv","p_channel_radio","p_channel_press","p_channel_event","p_channel_demo","p_channel_details","p_purpose","p_discount_active"],
    "reason": ["r_reason_sk","r_reason_id","r_reason_desc"],
    "ship_mode": ["sm_ship_mode_sk","sm_ship_mode_id","sm_type","sm_code","sm_carrier","sm_contract"],
    "store": ["s_store_sk","s_store_id","s_rec_start_date","s_rec_end_date","s_closed_date_sk","s_store_name","s_number_employees","s_floor_space","s_hours","s_manager","s_market_id","s_geography_class","s_market_desc","s_market_manager","s_division_id","s_division_name","s_company_id","s_company_name","s_street_number","s_street_name","s_street_type","s_suite_number","s_city","s_county","s_state","s_zip","s_country","s_gmt_offset","s_tax_precentage"],
    "store_returns": ["sr_returned_date_sk","sr_return_time_sk","sr_item_sk","sr_customer_sk","sr_cdemo_sk","sr_hdemo_sk","sr_addr_sk","sr_store_sk","sr_reason_sk","sr_ticket_number","sr_return_quantity","sr_return_amt","sr_return_tax","sr_return_amt_inc_tax","sr_fee","sr_return_ship_cost","sr_refunded_cash","sr_reversed_charge","sr_store_credit","sr_net_loss"],
    "store_sales": ["ss_sold_date_sk","ss_sold_time_sk","ss_item_sk","ss_customer_sk","ss_cdemo_sk","ss_hdemo_sk","ss_addr_sk","ss_store_sk","ss_promo_sk","ss_ticket_number","ss_quantity","ss_wholesale_cost","ss_list_price","ss_sales_price","ss_ext_discount_amt","ss_ext_sales_price","ss_ext_wholesale_cost","ss_ext_list_price","ss_ext_tax","ss_coupon_amt","ss_net_paid","ss_net_paid_inc_tax","ss_net_profit"],
    "time_dim": ["t_time_sk","t_time_id","t_time","t_hour","t_minute","t_second","t_am_pm","t_shift","t_sub_shift","t_meal_time"],
    "warehouse": ["w_warehouse_sk","w_warehouse_id","w_warehouse_name","w_warehouse_sq_ft","w_street_number","w_street_name","w_street_type","w_suite_number","w_city","w_county","w_state","w_zip","w_country","w_gmt_offset"],
    "web_page": ["wp_web_page_sk","wp_web_page_id","wp_rec_start_date","wp_rec_end_date","wp_creation_date_sk","wp_access_date_sk","wp_autogen_flag","wp_customer_sk","wp_url","wp_type","wp_char_count","wp_link_count","wp_image_count","wp_max_ad_count"],
    "web_returns": ["wr_returned_date_sk","wr_returned_time_sk","wr_item_sk","wr_refunded_customer_sk","wr_refunded_cdemo_sk","wr_refunded_hdemo_sk","wr_refunded_addr_sk","wr_returning_customer_sk","wr_returning_cdemo_sk","wr_returning_hdemo_sk","wr_returning_addr_sk","wr_web_page_sk","wr_reason_sk","wr_order_number","wr_return_quantity","wr_return_amt","wr_return_tax","wr_return_amt_inc_tax","wr_fee","wr_return_ship_cost","wr_refunded_cash","wr_reversed_charge","wr_account_credit","wr_net_loss"],
    # ATENÇÃO (fix 2026-08-19): ordem DEVE ser a do .dat do dsdgen (spec TPC-DS,
    # ws_item_sk na posição 4; web_page antes de web_site). A versão anterior
    # tinha ws_item_sk no fim + web_site/web_page trocados → rename posicional
    # desalinhou TODAS as colunas a partir da 4ª em todos os Silvers gerados
    # (ws_order_number continha promo_sk → q95 explodia em join quase-cartesiano).
    "web_sales": ["ws_sold_date_sk","ws_sold_time_sk","ws_ship_date_sk","ws_item_sk","ws_bill_customer_sk","ws_bill_cdemo_sk","ws_bill_hdemo_sk","ws_bill_addr_sk","ws_ship_customer_sk","ws_ship_cdemo_sk","ws_ship_hdemo_sk","ws_ship_addr_sk","ws_web_page_sk","ws_web_site_sk","ws_ship_mode_sk","ws_warehouse_sk","ws_promo_sk","ws_order_number","ws_quantity","ws_wholesale_cost","ws_list_price","ws_sales_price","ws_ext_discount_amt","ws_ext_sales_price","ws_ext_wholesale_cost","ws_ext_list_price","ws_ext_tax","ws_coupon_amt","ws_ext_ship_cost","ws_net_paid","ws_net_paid_inc_tax","ws_net_paid_inc_ship","ws_net_paid_inc_ship_tax","ws_net_profit"],
    "web_site": ["web_site_sk","web_site_id","web_rec_start_date","web_rec_end_date","web_name","web_open_date_sk","web_close_date_sk","web_class","web_manager","web_mkt_id","web_mkt_class","web_mkt_desc","web_market_manager","web_company_id","web_company_name","web_street_number","web_street_name","web_street_type","web_suite_number","web_city","web_county","web_state","web_zip","web_country","web_gmt_offset","web_tax_percentage"],
}


def get_system_info():
    """Collect system information for documentation."""
    return {
        "hostname": platform.node(),
        "os": platform.system(),
        "os_version": platform.release(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
    }


def get_spark_config_info(spark: SparkSession) -> dict:
    """Extract Spark configuration for documentation."""
    conf = spark.sparkContext.getConf()
    return {
        "spark_version": spark.version,
        "app_name": conf.get("spark.app.name", "unknown"),
        "master": conf.get("spark.master", "unknown"),
        "executor_memory": conf.get("spark.executor.memory", "unknown"),
        "executor_cores": conf.get("spark.executor.cores", "unknown"),
        "driver_memory": conf.get("spark.driver.memory", "unknown"),
    }


def save_metrics(metrics: dict, filename: str):
    """Save metrics to JSON file."""
    os.makedirs(METRICS_PATH, exist_ok=True)
    filepath = os.path.join(METRICS_PATH, filename)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to: {filepath}")


def get_spark_session() -> SparkSession:
    """Create and configure SparkSession for Delta Lake."""
    
    builder = (
        SparkSession.builder
        .appName(f"Bronze to Silver - SF{SCALE_FACTOR} - {LAKEHOUSE_TYPE}")
        # S3/MinIO configuration
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        # Delta Lake configuration
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # Performance
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    )
    
    return builder.getOrCreate()


def convert_table(spark: SparkSession, table_name: str, bronze_path: str, silver_path: str, s3=None) -> dict:
    """Convert a single table from Bronze to Silver layer.

    Collects per-table metrics: row_count, read_time, write_time, bronze_size,
    silver_size, delta_overhead_ratio, num_output_files.
    """

    start_time = time.time()

    try:
        # --- READ phase (counted via df.count() materialization) ---
        read_start = time.time()
        df = spark.read.parquet(f"{bronze_path}/{table_name}")

        # Fix generic column names (column0, column1, ...) from dsdgen-generated Parquet
        current_cols = df.columns
        if current_cols and current_cols[0].startswith("column") and table_name in TPCDS_COLUMNS:
            expected_cols = TPCDS_COLUMNS[table_name]
            if len(current_cols) >= len(expected_cols):
                # Select only the expected number of columns (drop extras from schema contamination)
                df = df.select(current_cols[:len(expected_cols)])
                for old_name, new_name in zip(current_cols[:len(expected_cols)], expected_cols):
                    df = df.withColumnRenamed(old_name, new_name)
                if len(current_cols) > len(expected_cols):
                    print(f"(trimmed {len(current_cols)}→{len(expected_cols)} cols, renamed) ", end="", flush=True)
                else:
                    print(f"(renamed {len(expected_cols)} columns) ", end="", flush=True)
            else:
                print(f"(WARNING: too few columns: {len(current_cols)} vs {len(expected_cols)}) ", end="", flush=True)

        # Typed cast when the benchmark declares column types (TPC-H). TPC-DS
        # declares none -> skipped, preserving the string-typed Silver.
        col_types = _BENCHMARK.column_types(table_name)
        if col_types:
            df = df.selectExpr(*cast_select_exprs(col_types))

        row_count = df.count()
        read_time = time.time() - read_start

        # --- WRITE phase ---
        write_start = time.time()
        output_path = f"{silver_path}/{table_name}"
        df.write.format("delta").mode("overwrite").save(output_path)
        write_time = time.time() - write_start

        elapsed_time = time.time() - start_time

        # --- SIZE collection (MinIO list) ---
        bronze_bytes = bronze_files = silver_bytes = silver_files = 0
        if s3 is not None:
            bronze_prefix = f"sf{SCALE_FACTOR}/{_BENCHMARK.bronze_segment}/{table_name}/"
            bronze_bytes, bronze_files = _s3_prefix_size(s3, "bronze", bronze_prefix)
            silver_prefix_obj = f"{_BENCHMARK.silver_key(SCALE_FACTOR, LAKEHOUSE_TYPE, FILE_FORMAT)}/{table_name}/"
            silver_bytes, silver_files = _s3_prefix_size(s3, "silver", silver_prefix_obj)

        overhead = (silver_bytes / bronze_bytes) if bronze_bytes > 0 else 0

        return {
            "table_name": table_name,
            "row_count": row_count,
            "read_time_seconds": round(read_time, 2),
            "write_time_seconds": round(write_time, 2),
            "conversion_time_seconds": round(elapsed_time, 2),
            "bronze_size_mb": round(bronze_bytes / (1024 ** 2), 2),
            "silver_size_mb": round(silver_bytes / (1024 ** 2), 2),
            "silver_num_files": silver_files,
            "delta_overhead_ratio": round(overhead, 3),
            "status": "success",
            "output_path": output_path,
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "table_name": table_name,
            "row_count": 0,
            "conversion_time_seconds": round(elapsed_time, 2),
            "status": "failed",
            "error": str(e),
        }


def main():
    print("=" * 60)
    print(f"Bronze to Silver Conversion")
    print(f"Scale Factor: {SCALE_FACTOR}")
    print(f"Lakehouse Type: {LAKEHOUSE_TYPE}")
    print(f"File Format: {FILE_FORMAT}")
    print(f"MinIO Endpoint: {MINIO_ENDPOINT}")
    print("=" * 60)
    
    # Paths
    bronze_path = f"s3a://bronze/sf{SCALE_FACTOR}/{_BENCHMARK.bronze_segment}"
    silver_path = f"s3a://silver/{_BENCHMARK.silver_key(SCALE_FACTOR, LAKEHOUSE_TYPE, FILE_FORMAT)}"
    
    print(f"\nBronze Path: {bronze_path}")
    print(f"Silver Path: {silver_path}")
    
    # Create Spark session
    print("\nInitializing Spark session...")
    spark = get_spark_session()
    
    # Collect configuration info
    system_info = get_system_info()
    spark_config = get_spark_config_info(spark)
    
    print(f"Spark Version: {spark_config['spark_version']}")
    print(f"Executor Memory: {spark_config['executor_memory']}")
    
    # Convert tables
    start_time = time.time()
    table_metrics = []
    s3 = _s3_client()

    print(f"\nConverting {len(_BENCHMARK.tables)} tables...")

    for table_name in _BENCHMARK.tables:
        print(f"  Converting {table_name}...", end=" ", flush=True)
        result = convert_table(spark, table_name, bronze_path, silver_path, s3=s3)
        table_metrics.append(result)
        
        if result["status"] == "success":
            print(f"OK ({result['row_count']:,} rows, {result['conversion_time_seconds']:.2f}s)")
        else:
            print(f"FAILED: {result.get('error', 'Unknown error')}")
    
    total_time = time.time() - start_time
    
    # Summary
    successful = sum(1 for t in table_metrics if t["status"] == "success")
    total_rows = sum(t["row_count"] for t in table_metrics)
    
    print(f"\n{'=' * 60}")
    print(f"Conversion Summary:")
    print(f"  Tables: {successful}/{len(_BENCHMARK.tables)} successful")
    print(f"  Total Rows: {total_rows:,}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Output: {silver_path}")
    print(f"{'=' * 60}")
    
    # Save metrics
    total_bronze_mb = sum(t.get("bronze_size_mb", 0) for t in table_metrics)
    total_silver_mb = sum(t.get("silver_size_mb", 0) for t in table_metrics)
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "scale_factor": SCALE_FACTOR,
        "lakehouse_type": LAKEHOUSE_TYPE,
        "file_format": FILE_FORMAT,
        "bronze_path": bronze_path,
        "silver_path": silver_path,
        "total_time_seconds": round(total_time, 2),
        "total_rows": total_rows,
        "total_bronze_mb": round(total_bronze_mb, 2),
        "total_silver_mb": round(total_silver_mb, 2),
        "overall_delta_overhead_ratio": round(total_silver_mb / total_bronze_mb, 3) if total_bronze_mb > 0 else 0,
        "tables_successful": successful,
        "tables_total": len(_BENCHMARK.tables),
        "system_info": system_info,
        "spark_config": spark_config,
        "tables": table_metrics,
    }
    filename = f"silver_conversion_sf{SCALE_FACTOR}_{LAKEHOUSE_TYPE}_{FILE_FORMAT}_{int(start_time)}.json"
    save_metrics(metrics, filename)

    # Upload metrics JSON to MinIO so it survives pod deletion.
    s3_up = _s3_client()
    if s3_up is not None:
        try:
            key = f"{METRICS_PREFIX}/{filename}"
            s3_up.put_object(
                Bucket=METRICS_BUCKET,
                Key=key,
                Body=json.dumps(metrics, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
            print(f"  Uploaded metrics to s3://{METRICS_BUCKET}/{key}")
        except Exception as e:
            print(f"  WARN: failed to upload metrics to MinIO: {e}")

    spark.stop()
    
    if successful < len(_BENCHMARK.tables):
        print("\nWARNING: Some tables failed to convert!")
        sys.exit(1)
    
    print("\n✓ Silver layer conversion complete!")


if __name__ == "__main__":
    main()
