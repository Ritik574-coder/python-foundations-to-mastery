"""
Pytest fixtures for the E-Commerce Lakehouse Platform tests.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    return SparkSession.builder \
        .master("local[*]") \
        .appName("ECommerceLakehouseTests") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()


@pytest.fixture
def sample_orders(spark):
    """Sample orders DataFrame."""
    data = [
        ("ORD001", "C001", "P101", 2, 29.99, "2024-01-10", "COMPLETED", "Credit Card"),
        ("ORD002", "C002", "P102", 1, 49.99, "2024-01-12", "COMPLETED", "PayPal"),
        ("ORD003", "C003", "P103", 3, 19.99, "2024-01-15", "COMPLETED", "Debit Card"),
        ("ORD004", "C001", "P104", 1, 99.99, "2024-01-18", "PENDING", "Credit Card"),
        ("ORD005", "C004", "P105", 2, 34.99, "2024-01-20", "COMPLETED", "PayPal"),
    ]
    
    columns = ["order_id", "customer_id", "product_id", "quantity", "unit_price", 
               "order_date", "status", "payment_method"]
    
    return spark.createDataFrame(data, columns)


@pytest.fixture
def sample_customers(spark):
    """Sample customers DataFrame."""
    data = [
        ("C001", "John Smith", "john@email.com", "555-0101", "New York", "NY", "USA", "2023-01-15", "Premium"),
        ("C002", "Jane Doe", "jane@email.com", "555-0102", "Los Angeles", "CA", "USA", "2023-02-20", "Standard"),
        ("C003", "Bob Johnson", "bob@email.com", "555-0103", "Chicago", "IL", "USA", "2023-03-10", "Premium"),
        ("C004", "Alice Brown", "alice@email.com", "555-0104", "Houston", "TX", "USA", "2023-04-05", "Standard"),
    ]
    
    columns = ["customer_id", "name", "email", "phone", "city", "state", "country", 
               "signup_date", "segment"]
    
    return spark.createDataFrame(data, columns)


@pytest.fixture
def sample_products(spark):
    """Sample products DataFrame."""
    data = [
        ("P101", "Wireless Mouse", "Electronics", "TechBrand", 29.99, 12.50, 500, "2023-01-01"),
        ("P102", "Bluetooth Keyboard", "Electronics", "TechBrand", 49.99, 22.00, 300, "2023-01-15"),
        ("P103", "USB-C Hub", "Electronics", "ConnectPro", 19.99, 8.50, 1000, "2023-02-01"),
        ("P104", "4K Monitor", "Electronics", "DisplayMax", 99.99, 65.00, 150, "2023-02-15"),
        ("P105", "Webcam HD", "Electronics", "ClearView", 34.99, 15.00, 400, "2023-03-01"),
    ]
    
    columns = ["product_id", "product_name", "category", "brand", "price", "cost", 
               "stock_quantity", "created_date"]
    
    return spark.createDataFrame(data, columns)


@pytest.fixture
def sample_payments(spark):
    """Sample payments DataFrame."""
    data = [
        ("PAY001", "ORD001", 59.98, "2024-01-10", "Credit Card", "Completed", 1.80),
        ("PAY002", "ORD002", 49.99, "2024-01-12", "PayPal", "Completed", 1.50),
        ("PAY003", "ORD003", 59.97, "2024-01-15", "Debit Card", "Completed", 0.90),
        ("PAY004", "ORD005", 69.98, "2024-01-20", "PayPal", "Completed", 2.10),
    ]
    
    columns = ["payment_id", "order_id", "amount", "payment_date", "payment_method", 
               "status", "transaction_fee"]
    
    return spark.createDataFrame(data, columns)
