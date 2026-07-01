"""
Daily pipeline entry point for the E-Commerce Lakehouse Platform.
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import SparkSession

from src.config.settings import AppConfig
from src.utils.spark import get_spark_session
from src.bronze.ingest import run_bronze_ingestion, BronzeIngestion
from src.silver.orders import process_silver_orders
from src.gold.revenue import process_gold_revenue
from src.utils.validation import DataValidator


def run_daily_pipeline(execution_date: str = None, environment: str = "dev") -> None:
    """
    Run the daily ETL pipeline.
    
    Args:
        execution_date: Execution date (YYYY-MM-DD format)
        environment: Environment (dev/test/prod)
    """
    print("=" * 60)
    print("E-Commerce Lakehouse Platform - Daily Pipeline")
    print("=" * 60)
    print(f"Execution Date: {execution_date or datetime.now().strftime('%Y-%m-%d')}")
    print(f"Environment: {environment}")
    print("=" * 60)
    
    # Initialize configuration and Spark
    config = AppConfig(environment=environment)
    config.pipeline.execution_date = execution_date
    
    spark = get_spark_session(config)
    
    try:
        # Step 1: Bronze Layer Ingestion
        print("\n1. Bronze Layer Ingestion")
        print("-" * 40)
        
        source_paths = {
            "orders": os.path.join(config.storage.base_path, "raw", "orders.csv"),
            "customers": os.path.join(config.storage.base_path, "raw", "customers.csv"),
            "products": os.path.join(config.storage.base_path, "raw", "products.csv"),
            "payments": os.path.join(config.storage.base_path, "raw", "payments.csv")
        }
        
        bronze_results = run_bronze_ingestion(spark, config, execution_date)
        
        # Step 2: Silver Layer Processing
        print("\n2. Silver Layer Processing")
        print("-" * 40)
        
        bronze_orders_path = f"{config.storage.get_layer_path('bronze')}/orders"
        bronze_payments_path = f"{config.storage.get_layer_path('bronze')}/payments"
        bronze_products_path = f"{config.storage.get_layer_path('bronze')}/products"
        
        # Process orders
        silver_orders = process_silver_orders(
            spark, config,
            bronze_orders_path,
            bronze_payments_path
        )
        print(f"✓ Processed {silver_orders.count()} orders")
        
        # Step 3: Gold Layer Aggregations
        print("\n3. Gold Layer Aggregations")
        print("-" * 40)
        
        silver_orders_path = f"{config.storage.get_layer_path('silver')}/orders"
        silver_payments_path = f"{config.storage.get_layer_path('silver')}/payments"
        silver_products_path = f"{config.storage.get_layer_path('silver')}/products"
        
        process_gold_revenue(
            spark, config,
            silver_orders_path,
            silver_payments_path,
            silver_products_path
        )
        
        # Step 4: Data Quality Validation
        print("\n4. Data Quality Validation")
        print("-" * 40)
        
        validator = DataValidator()
        
        # Validate silver orders
        silver_orders_df = spark.read.format("delta").load(silver_orders_path)
        validator.validate_completeness(silver_orders_df, "order_id")
        validator.validate_completeness(silver_orders_df, "customer_id")
        validator.validate_uniqueness(silver_orders_df, ["order_id"])
        validator.validate_range(silver_orders_df, "amount", min_value=0)
        
        validator.print_summary()
        
        # Pipeline summary
        print("\n" + "=" * 60)
        print("Pipeline Completed Successfully")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {str(e)}")
        raise
    
    finally:
        spark.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="E-Commerce Lakehouse Daily Pipeline")
    parser.add_argument("--execution-date", type=str, help="Execution date (YYYY-MM-DD)")
    parser.add_argument("--environment", type=str, default="dev", choices=["dev", "test", "prod"])
    
    args = parser.parse_args()
    
    run_daily_pipeline(
        execution_date=args.execution_date,
        environment=args.environment
    )
