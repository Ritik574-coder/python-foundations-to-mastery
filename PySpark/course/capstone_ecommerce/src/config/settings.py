"""
Configuration settings for the E-Commerce Lakehouse Platform.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class StorageConfig:
    """Storage configuration."""
    base_path: str = "s3://data-lake/ecommerce"
    bronze_path: str = "bronze"
    silver_path: str = "silver"
    gold_path: str = "gold"
    
    def get_layer_path(self, layer: str) -> str:
        return f"{self.base_path}/{layer}"


@dataclass
class SparkConfig:
    """Spark configuration."""
    app_name: str = "ECommerceLakehouse"
    master: str = "local[*]"
    shuffle_partitions: int = 200
    driver_memory: str = "4g"
    executor_memory: str = "8g"
    adaptive_enabled: bool = True


@dataclass
class DataQualityConfig:
    """Data quality configuration."""
    quarantine_path: str = "quarantine"
    max_null_percentage: float = 0.05
    required_columns: dict = None
    
    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = {
                "orders": ["order_id", "customer_id", "order_date", "amount"],
                "customers": ["customer_id", "name", "email"],
                "products": ["product_id", "product_name", "price"],
                "payments": ["payment_id", "order_id", "amount", "payment_date"]
            }


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    execution_date: Optional[str] = None
    backfill_start: Optional[str] = None
    backfill_end: Optional[str] = None
    enable_validation: bool = True
    enable_quarantine: bool = True


class AppConfig:
    """Main application configuration."""
    
    def __init__(self, environment: str = "dev"):
        self.environment = environment
        self.storage = StorageConfig()
        self.spark = SparkConfig()
        self.data_quality = DataQualityConfig()
        self.pipeline = PipelineConfig()
        
        # Override with environment-specific settings
        self._apply_environment_config()
    
    def _apply_environment_config(self):
        """Apply environment-specific configurations."""
        if self.environment == "dev":
            self.spark.master = "local[*]"
            self.spark.shuffle_partitions = 8
            self.spark.driver_memory = "2g"
            self.spark.executor_memory = "2g"
            self.storage.base_path = "/tmp/data-lake/ecommerce"
        elif self.environment == "test":
            self.spark.master = "local[*]"
            self.spark.shuffle_partitions = 4
            self.spark.driver_memory = "1g"
            self.spark.executor_memory = "1g"
            self.storage.base_path = "/tmp/test-lake/ecommerce"
        elif self.environment == "prod":
            self.spark.shuffle_partitions = 200
            self.spark.driver_memory = "4g"
            self.spark.executor_memory = "8g"
