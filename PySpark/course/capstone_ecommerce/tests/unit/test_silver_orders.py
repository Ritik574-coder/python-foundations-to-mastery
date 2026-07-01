"""
Unit tests for silver orders processor.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from src.silver.orders import OrdersSilverProcessor
from src.config.settings import AppConfig


class TestOrdersSilverProcessor:
    """Test cases for OrdersSilverProcessor."""
    
    def test_clean_orders(self, spark, sample_orders):
        """Test order cleaning transformations."""
        config = AppConfig(environment="test")
        processor = OrdersSilverProcessor(spark, config)
        
        # Process orders
        result = processor.process(sample_orders)
        
        # Verify cleaning
        assert result.count() == sample_orders.count()
        assert "order_id" in result.columns
        assert "amount" in result.columns
        
        # Verify amount calculation
        first_row = result.first()
        assert first_row.amount == first_row.quantity * first_row.unit_price
    
    def test_validate_orders(self, spark, sample_orders):
        """Test order validation."""
        config = AppConfig(environment="test")
        processor = OrdersSilverProcessor(spark, config)
        
        # Add invalid records
        invalid_data = [
            (None, "C001", "P101", 1, 10.00, "2024-01-01", "COMPLETED", "Credit Card"),  # Missing order_id
            ("ORD006", None, "P101", 1, 10.00, "2024-01-01", "COMPLETED", "Credit Card"),  # Missing customer_id
            ("ORD007", "C001", "P101", 1, -10.00, "2024-01-01", "COMPLETED", "Credit Card"),  # Negative amount
            ("ORD008", "C001", "P101", 1, 10.00, "2024-01-01", "INVALID", "Credit Card"),  # Invalid status
        ]
        
        invalid_columns = ["order_id", "customer_id", "product_id", "quantity", "unit_price", 
                          "order_date", "status", "payment_method"]
        
        invalid_df = spark.createDataFrame(invalid_data, invalid_columns)
        combined_df = sample_orders.union(invalid_df)
        
        # Validate
        valid_df, invalid_df = processor._validate_orders(combined_df)
        
        # Verify separation
        assert valid_df.count() == sample_orders.count()
        assert invalid_df.count() == 4
    
    def test_status_normalization(self, spark):
        """Test status normalization to uppercase."""
        config = AppConfig(environment="test")
        processor = OrdersSilverProcessor(spark, config)
        
        data = [
            ("ORD001", "C001", "P101", 1, 10.00, "2024-01-01", "completed", "Credit Card"),
            ("ORD002", "C002", "P102", 1, 20.00, "2024-01-01", "  Pending  ", "PayPal"),
        ]
        
        columns = ["order_id", "customer_id", "product_id", "quantity", "unit_price", 
                   "order_date", "status", "payment_method"]
        
        df = spark.createDataFrame(data, columns)
        result = processor._clean_orders(df)
        
        # Verify uppercase status
        statuses = [row.status for row in result.collect()]
        assert all(s == s.upper() for s in statuses)


class TestRevenueMetrics:
    """Test cases for revenue metrics."""
    
    def test_calculate_daily_revenue(self, spark, sample_orders, sample_payments):
        """Test daily revenue calculation."""
        from src.gold.revenue import RevenueMetrics
        
        config = AppConfig(environment="test")
        metrics = RevenueMetrics(spark, config)
        
        # Process orders first
        processor = OrdersSilverProcessor(spark, config)
        silver_orders = processor.process(sample_orders)
        
        # Calculate daily revenue
        daily_revenue = metrics.calculate_daily_revenue(silver_orders, sample_payments)
        
        # Verify
        assert daily_revenue.count() > 0
        assert "date" in daily_revenue.columns
        assert "total_revenue" in daily_revenue.columns
    
    def test_calculate_product_metrics(self, spark, sample_orders, sample_products):
        """Test product metrics calculation."""
        from src.gold.revenue import RevenueMetrics
        
        config = AppConfig(environment="test")
        metrics = RevenueMetrics(spark, config)
        
        # Process orders first
        processor = OrdersSilverProcessor(spark, config)
        silver_orders = processor.process(sample_orders)
        
        # Calculate product metrics
        product_metrics = metrics.calculate_product_metrics(silver_orders, sample_products)
        
        # Verify
        assert product_metrics.count() > 0
        assert "product_id" in product_metrics.columns
        assert "total_revenue" in product_metrics.columns
