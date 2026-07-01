"""
Silver layer transformations for orders.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, trim, upper, to_date, current_date, lit
)
from pyspark.sql.types import DoubleType

from src.config.settings import AppConfig
from src.io.writers import DataWriter, QuarantineWriter


class OrdersSilverProcessor:
    """Process orders data for silver layer."""
    
    def __init__(self, spark: SparkSession, config: AppConfig):
        self.spark = spark
        self.config = config
        self.writer = DataWriter()
        self.quarantine_writer = QuarantineWriter()
    
    def process(self, bronze_orders: DataFrame, bronze_payments: DataFrame = None) -> DataFrame:
        """
        Process orders for silver layer.
        
        Args:
            bronze_orders: Raw orders DataFrame
            bronze_payments: Optional payments DataFrame for enrichment
            
        Returns:
            Cleaned orders DataFrame
        """
        # Step 1: Basic cleaning
        cleaned_df = self._clean_orders(bronze_orders)
        
        # Step 2: Validate and quarantine invalid records
        valid_df, invalid_df = self._validate_orders(cleaned_df)
        
        # Step 3: Quarantine invalid records
        if invalid_df.count() > 0:
            self._quarantine_invalid(invalid_df)
        
        # Step 4: Enrich with payments if available
        if bronze_payments:
            valid_df = self._enrich_with_payments(valid_df, bronze_payments)
        
        # Step 5: Add silver layer metadata
        final_df = valid_df \
            .withColumn("_processed_date", current_date()) \
            .withColumn("_layer", lit("silver"))
        
        return final_df
    
    def _clean_orders(self, df: DataFrame) -> DataFrame:
        """Clean orders data."""
        return df \
            .withColumn("order_id", trim(col("order_id"))) \
            .withColumn("customer_id", trim(col("customer_id"))) \
            .withColumn("status", upper(trim(col("status")))) \
            .withColumn("order_date", to_date(col("order_date"), "yyyy-MM-dd")) \
            .withColumn("amount", (col("quantity") * col("unit_price")).cast(DoubleType()))
    
    def _validate_orders(self, df: DataFrame):
        """Validate orders and separate valid/invalid records."""
        # Define validation rules
        valid_df = df.filter(
            col("order_id").isNotNull() &
            (col("order_id") != "") &
            col("customer_id").isNotNull() &
            (col("customer_id") != "") &
            col("order_date").isNotNull() &
            col("amount").isNotNull() &
            (col("amount") > 0) &
            col("status").isin("COMPLETED", "PENDING", "CANCELLED")
        )
        
        invalid_df = df.filter(
            col("order_id").isNull() |
            (col("order_id") == "") |
            col("customer_id").isNull() |
            (col("customer_id") == "") |
            col("order_date").isNull() |
            col("amount").isNull() |
            (col("amount") <= 0) |
            ~col("status").isin("COMPLETED", "PENDING", "CANCELLED")
        )
        
        return valid_df, invalid_df
    
    def _quarantine_invalid(self, invalid_df: DataFrame):
        """Write invalid records to quarantine."""
        quarantine_path = f"{self.config.storage.get_layer_path('quarantine')}/orders"
        
        self.quarantine_writer.write_quarantine(
            invalid_df,
            quarantine_path,
            source_name="orders",
            error_message="Validation failed: missing required fields or invalid values"
        )
    
    def _enrich_with_payments(self, orders_df: DataFrame, payments_df: DataFrame) -> DataFrame:
        """Enrich orders with payment information."""
        # Aggregate payments by order
        payments_agg = payments_df \
            .groupBy("order_id") \
            .agg(
                {"amount": "sum", "transaction_fee": "sum"}
            ) \
            .withColumnRenamed("sum(amount)", "total_payment") \
            .withColumnRenamed("sum(transaction_fee)", "total_fee")
        
        # Join with orders
        enriched_df = orders_df.join(
            payments_agg,
            orders_df.order_id == payments_agg.order_id,
            "left"
        ).drop(payments_agg.order_id)
        
        return enriched_df


def process_silver_orders(spark: SparkSession, config: AppConfig,
                          bronze_orders_path: str, bronze_payments_path: str = None) -> DataFrame:
    """
    Process orders for silver layer.
    
    Args:
        spark: SparkSession
        config: Application configuration
        bronze_orders_path: Path to bronze orders
        bronze_payments_path: Path to bronze payments (optional)
        
    Returns:
        Processed orders DataFrame
    """
    # Read bronze data
    orders_df = spark.read.parquet(bronze_orders_path)
    payments_df = spark.read.parquet(bronze_payments_path) if bronze_payments_path else None
    
    # Process
    processor = OrdersSilverProcessor(spark, config)
    silver_orders = processor.process(orders_df, payments_df)
    
    # Write to silver
    silver_path = f"{config.storage.get_layer_path('silver')}/orders"
    processor.writer.write_delta(silver_orders, silver_path, mode="overwrite")
    
    return silver_orders
