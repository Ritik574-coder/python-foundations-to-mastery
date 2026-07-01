"""
Spark session utilities for the E-Commerce Lakehouse Platform.
"""

from pyspark.sql import SparkSession
from src.config.settings import AppConfig


def get_spark_session(config: AppConfig) -> SparkSession:
    """
    Create a SparkSession with the given configuration.
    
    Args:
        config: Application configuration
        
    Returns:
        Configured SparkSession
    """
    builder = SparkSession.builder \
        .appName(config.spark.app_name) \
        .master(config.spark.master) \
        .config("spark.sql.shuffle.partitions", str(config.spark.shuffle_partitions)) \
        .config("spark.driver.memory", config.spark.driver_memory) \
        .config("spark.executor.memory", config.spark.executor_memory) \
        .config("spark.sql.adaptive.enabled", str(config.spark.adaptive_enabled))
    
    # Add Delta Lake configuration if needed
    if config.environment == "prod":
        builder = builder \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    
    return builder.getOrCreate()
