"""
Bronze layer ingestion for the E-Commerce Lakehouse Platform.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import current_date, input_file_name, lit
from typing import Dict, List

from src.io.readers import EcommerceDataReader
from src.io.writers import DataWriter
from src.config.settings import AppConfig


class BronzeIngestion:
    """Bronze layer ingestion handler."""
    
    def __init__(self, spark: SparkSession, config: AppConfig):
        self.spark = spark
        self.config = config
        self.reader = EcommerceDataReader(spark)
        self.writer = DataWriter()
    
    def ingest_all(self, source_paths: Dict[str, str]) -> Dict[str, bool]:
        """
        Ingest all data sources to bronze layer.
        
        Args:
            source_paths: Dictionary mapping table names to source paths
            
        Returns:
            Dictionary of ingestion results
        """
        results = {}
        
        for table_name, source_path in source_paths.items():
            try:
                self.ingest_table(table_name, source_path)
                results[table_name] = True
                print(f"✓ Successfully ingested {table_name}")
            except Exception as e:
                results[table_name] = False
                print(f"✗ Failed to ingest {table_name}: {str(e)}")
        
        return results
    
    def ingest_table(self, table_name: str, source_path: str) -> None:
        """
        Ingest a single table to bronze layer.
        
        Args:
            table_name: Name of the table
            source_path: Path to source data
        """
        # Determine format from file extension
        if source_path.endswith(".csv"):
            format_type = "csv"
        elif source_path.endswith(".json"):
            format_type = "json"
        elif source_path.endswith(".parquet"):
            format_type = "parquet"
        else:
            format_type = "csv"
        
        # Read source data with metadata
        df = self.reader.read_with_metadata(source_path, format_type)
        
        # Add ingestion metadata
        df_with_meta = df \
            .withColumn("_ingestion_date", current_date()) \
            .withColumn("_table_name", lit(table_name))
        
        # Write to bronze layer
        bronze_path = f"{self.config.storage.get_layer_path('bronze')}/{table_name}"
        
        self.writer.write_parquet(
            df_with_meta,
            bronze_path,
            mode="append",
            partition_cols=["_ingestion_date"]
        )
        
        # Log metrics
        record_count = df_with_meta.count()
        print(f"  Ingested {record_count} records to {bronze_path}")
    
    def validate_bronze(self, table_name: str) -> Dict[str, int]:
        """
        Validate bronze layer data.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with validation metrics
        """
        bronze_path = f"{self.config.storage.get_layer_path('bronze')}/{table_name}"
        
        try:
            df = self.spark.read.parquet(bronze_path)
            
            metrics = {
                "total_records": df.count(),
                "total_columns": len(df.columns),
                "null_counts": {
                    col: df.filter(df[col].isNull()).count()
                    for col in df.columns
                }
            }
            
            return metrics
            
        except Exception as e:
            return {"error": str(e)}


def run_bronze_ingestion(spark: SparkSession, config: AppConfig, 
                         execution_date: str = None) -> Dict[str, bool]:
    """
    Run bronze layer ingestion.
    
    Args:
        spark: SparkSession
        config: Application configuration
        execution_date: Execution date (optional)
        
    Returns:
        Dictionary of ingestion results
    """
    # Define source paths
    source_paths = {
        "orders": f"{config.storage.base_path}/raw/orders.csv",
        "customers": f"{config.storage.base_path}/raw/customers.csv",
        "products": f"{config.storage.base_path}/raw/products.csv",
        "payments": f"{config.storage.base_path}/raw/payments.csv"
    }
    
    # Run ingestion
    ingestion = BronzeIngestion(spark, config)
    results = ingestion.ingest_all(source_paths)
    
    return results
