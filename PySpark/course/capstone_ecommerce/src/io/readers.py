"""
Data readers for the E-Commerce Lakehouse Platform.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
from pyspark.sql.functions import input_file_name, current_timestamp, lit


class DataReader:
    """Base data reader class."""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
    
    def read_csv(self, path: str, schema: StructType = None, header: bool = True) -> DataFrame:
        """Read CSV file with optional schema."""
        reader = self.spark.read.option("header", str(header).lower())
        
        if schema:
            reader = reader.schema(schema)
        else:
            reader = reader.option("inferSchema", "true")
        
        return reader.csv(path)
    
    def read_json(self, path: str, schema: StructType = None) -> DataFrame:
        """Read JSON file."""
        reader = self.spark.read
        
        if schema:
            reader = reader.schema(schema)
        
        return reader.json(path)
    
    def read_parquet(self, path: str) -> DataFrame:
        """Read Parquet file."""
        return self.spark.read.parquet(path)
    
    def read_delta(self, path: str) -> DataFrame:
        """Read Delta table."""
        return self.spark.read.format("delta").load(path)


class EcommerceDataReader(DataReader):
    """E-commerce specific data readers."""
    
    def __init__(self, spark: SparkSession):
        super().__init__(spark)
        self._define_schemas()
    
    def _define_schemas(self):
        """Define schemas for e-commerce data."""
        self.schemas = {
            "orders": StructType([
                StructField("order_id", StringType(), False),
                StructField("customer_id", StringType(), False),
                StructField("product_id", StringType(), True),
                StructField("quantity", IntegerType(), True),
                StructField("unit_price", DoubleType(), True),
                StructField("order_date", StringType(), True),
                StructField("status", StringType(), True),
                StructField("payment_method", StringType(), True)
            ]),
            "customers": StructType([
                StructField("customer_id", StringType(), False),
                StructField("name", StringType(), False),
                StructField("email", StringType(), True),
                StructField("phone", StringType(), True),
                StructField("city", StringType(), True),
                StructField("state", StringType(), True),
                StructField("country", StringType(), True),
                StructField("signup_date", StringType(), True),
                StructField("segment", StringType(), True)
            ]),
            "products": StructType([
                StructField("product_id", StringType(), False),
                StructField("product_name", StringType(), False),
                StructField("category", StringType(), True),
                StructField("brand", StringType(), True),
                StructField("price", DoubleType(), True),
                StructField("cost", DoubleType(), True),
                StructField("stock_quantity", IntegerType(), True),
                StructField("created_date", StringType(), True)
            ]),
            "payments": StructType([
                StructField("payment_id", StringType(), False),
                StructField("order_id", StringType(), False),
                StructField("amount", DoubleType(), True),
                StructField("payment_date", StringType(), True),
                StructField("payment_method", StringType(), True),
                StructField("status", StringType(), True),
                StructField("transaction_fee", DoubleType(), True)
            ])
        }
    
    def read_orders(self, path: str) -> DataFrame:
        """Read orders data."""
        return self.read_csv(path, schema=self.schemas["orders"])
    
    def read_customers(self, path: str) -> DataFrame:
        """Read customers data."""
        return self.read_csv(path, schema=self.schemas["customers"])
    
    def read_products(self, path: str) -> DataFrame:
        """Read products data."""
        return self.read_csv(path, schema=self.schemas["products"])
    
    def read_payments(self, path: str) -> DataFrame:
        """Read payments data."""
        return self.read_csv(path, schema=self.schemas["payments"])
    
    def read_clickstream(self, path: str) -> DataFrame:
        """Read clickstream data."""
        return self.read_json(path)
    
    def read_with_metadata(self, path: str, format: str = "csv") -> DataFrame:
        """Read data with ingestion metadata."""
        if format == "csv":
            df = self.read_csv(path)
        elif format == "json":
            df = self.read_json(path)
        elif format == "parquet":
            df = self.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return df \
            .withColumn("_ingestion_timestamp", current_timestamp()) \
            .withColumn("_source_file", input_file_name())
