"""
Gold layer aggregations for revenue metrics.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg, min as spark_min, max as spark_max,
    countDistinct, current_date, lit, date_format
)
from pyspark.sql.window import Window

from src.config.settings import AppConfig
from src.io.writers import DataWriter


class RevenueMetrics:
    """Calculate revenue metrics for gold layer."""
    
    def __init__(self, spark: SparkSession, config: AppConfig):
        self.spark = spark
        self.config = config
        self.writer = DataWriter()
    
    def calculate_daily_revenue(self, orders_df: DataFrame, payments_df: DataFrame) -> DataFrame:
        """
        Calculate daily revenue metrics.
        
        Args:
            orders_df: Silver orders DataFrame
            payments_df: Silver payments DataFrame
            
        Returns:
            Daily revenue metrics DataFrame
        """
        # Join orders with payments
        enriched_df = orders_df.join(
            payments_df,
            "order_id",
            "inner"
        )
        
        # Calculate daily metrics
        daily_revenue = enriched_df \
            .filter(col("status") == "COMPLETED") \
            .groupBy(
                date_format("order_date", "yyyy-MM-dd").alias("date"),
                "payment_method"
            ) \
            .agg(
                count("order_id").alias("order_count"),
                countDistinct("customer_id").alias("unique_customers"),
                spark_sum("amount").alias("total_revenue"),
                spark_sum("transaction_fee").alias("total_fees"),
                avg("amount").alias("avg_order_value")
            ) \
            .withColumn("_processed_date", current_date())
        
        return daily_revenue
    
    def calculate_product_metrics(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """
        Calculate product performance metrics.
        
        Args:
            orders_df: Silver orders DataFrame
            products_df: Silver products DataFrame
            
        Returns:
            Product metrics DataFrame
        """
        # Join with products
        enriched_df = orders_df.join(
            products_df,
            "product_id",
            "inner"
        )
        
        # Calculate product metrics
        product_metrics = enriched_df \
            .filter(col("status") == "COMPLETED") \
            .groupBy(
                "product_id",
                "product_name",
                "category",
                "brand"
            ) \
            .agg(
                count("order_id").alias("order_count"),
                spark_sum("quantity").alias("total_quantity_sold"),
                spark_sum("amount").alias("total_revenue"),
                avg("amount").alias("avg_revenue_per_order"),
                countDistinct("customer_id").alias("unique_customers")
            ) \
            .withColumn("_processed_date", current_date())
        
        return product_metrics
    
    def calculate_customer_lifetime_value(self, orders_df: DataFrame) -> DataFrame:
        """
        Calculate customer lifetime value.
        
        Args:
            orders_df: Silver orders DataFrame
            
        Returns:
            Customer LTV DataFrame
        """
        # Calculate customer metrics
        customer_ltv = orders_df \
            .filter(col("status") == "COMPLETED") \
            .groupBy("customer_id") \
            .agg(
                count("order_id").alias("total_orders"),
                spark_sum("amount").alias("total_revenue"),
                avg("amount").alias("avg_order_value"),
                spark_min("order_date").alias("first_order_date"),
                spark_max("order_date").alias("last_order_date")
            ) \
            .withColumn("_processed_date", current_date())
        
        return customer_ltv
    
    def calculate_category_metrics(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """
        Calculate category performance metrics.
        
        Args:
            orders_df: Silver orders DataFrame
            products_df: Silver products DataFrame
            
        Returns:
            Category metrics DataFrame
        """
        # Join with products
        enriched_df = orders_df.join(
            products_df,
            "product_id",
            "inner"
        )
        
        # Calculate category metrics
        category_metrics = enriched_df \
            .filter(col("status") == "COMPLETED") \
            .groupBy("category") \
            .agg(
                count("order_id").alias("order_count"),
                spark_sum("amount").alias("total_revenue"),
                avg("amount").alias("avg_order_value"),
                countDistinct("product_id").alias("unique_products"),
                countDistinct("customer_id").alias("unique_customers")
            ) \
            .withColumn("_processed_date", current_date())
        
        return category_metrics


def process_gold_revenue(spark: SparkSession, config: AppConfig,
                         silver_orders_path: str, silver_payments_path: str,
                         silver_products_path: str) -> None:
    """
    Process gold layer revenue metrics.
    
    Args:
        spark: SparkSession
        config: Application configuration
        silver_orders_path: Path to silver orders
        silver_payments_path: Path to silver payments
        silver_products_path: Path to silver products
    """
    # Read silver data
    orders_df = spark.read.format("delta").load(silver_orders_path)
    payments_df = spark.read.format("delta").load(silver_payments_path)
    products_df = spark.read.format("delta").load(silver_products_path)
    
    # Initialize metrics calculator
    metrics = RevenueMetrics(spark, config)
    
    # Calculate and write metrics
    gold_path = config.storage.get_layer_path("gold")
    
    # Daily revenue
    daily_revenue = metrics.calculate_daily_revenue(orders_df, payments_df)
    metrics.writer.write_delta(daily_revenue, f"{gold_path}/daily_revenue", mode="overwrite")
    
    # Product metrics
    product_metrics = metrics.calculate_product_metrics(orders_df, products_df)
    metrics.writer.write_delta(product_metrics, f"{gold_path}/product_metrics", mode="overwrite")
    
    # Customer LTV
    customer_ltv = metrics.calculate_customer_lifetime_value(orders_df)
    metrics.writer.write_delta(customer_ltv, f"{gold_path}/customer_ltv", mode="overwrite")
    
    # Category metrics
    category_metrics = metrics.calculate_category_metrics(orders_df, products_df)
    metrics.writer.write_delta(category_metrics, f"{gold_path}/category_metrics", mode="overwrite")
    
    print("✓ Gold layer metrics calculated and written successfully")
