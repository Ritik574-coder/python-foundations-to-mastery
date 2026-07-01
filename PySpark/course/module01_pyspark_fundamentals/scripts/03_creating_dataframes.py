"""
Lesson 1.2.1: Creating and Inspecting DataFrames

This script demonstrates:
- Creating DataFrames from various sources
- Inspecting schema and data
- Basic DataFrame operations
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col, when


def main():
    spark = SparkSession.builder \
        .appName("Creating DataFrames") \
        .master("local[*]") \
        .getOrCreate()
    
    print("=" * 60)
    print("Creating and Inspecting DataFrames")
    print("=" * 60)
    
    # Method 1: From list of tuples
    print("\n1. Creating DataFrame from list of tuples:")
    data_tuples = [
        ("Alice", 25, 50000.0),
        ("Bob", 30, 60000.0),
        ("Charlie", 35, 75000.0),
        ("Diana", 28, 55000.0)
    ]
    
    df_tuples = spark.createDataFrame(data_tuples, ["name", "age", "salary"])
    df_tuples.show()
    
    # Method 2: From list of dictionaries
    print("\n2. Creating DataFrame from list of dictionaries:")
    data_dicts = [
        {"product": "Laptop", "price": 999.99, "quantity": 10},
        {"product": "Mouse", "price": 29.99, "quantity": 50},
        {"product": "Keyboard", "price": 49.99, "quantity": 30}
    ]
    
    df_dicts = spark.createDataFrame(data_dicts)
    df_dicts.show()
    
    # Method 3: With explicit schema
    print("\n3. Creating DataFrame with explicit schema:")
    schema = StructType([
        StructField("order_id", StringType(), False),
        StructField("customer", StringType(), False),
        StructField("amount", DoubleType(), False),
        StructField("quantity", IntegerType(), True)
    ])
    
    data_with_schema = [
        ("ORD001", "Alice", 150.00, 2),
        ("ORD002", "Bob", 75.50, 1),
        ("ORD003", "Charlie", 200.00, 3)
    ]
    
    df_schema = spark.createDataFrame(data_with_schema, schema)
    df_schema.show()
    
    # Inspecting DataFrames
    print("\n4. Inspecting DataFrames:")
    print("\n   a) Schema:")
    df_schema.printSchema()
    
    print("\n   b) First 5 rows (show):")
    df_schema.show(5)
    
    print("\n   c) Schema as string:")
    print(f"   {df_schema.schema.simpleString()}")
    
    print("\n   d) Column names:")
    print(f"   {df_schema.columns}")
    
    print("\n   e) Data types:")
    print(f"   {df_schema.dtypes}")
    
    print("\n   f) Count:")
    print(f"   {df_schema.count()}")
    
    print("\n   g) Describe (statistics):")
    df_schema.describe().show()
    
    # Reading from CSV file
    print("\n5. Reading from CSV file:")
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "sample_datasets")
    csv_path = os.path.join(data_dir, "customers.csv")
    
    if os.path.exists(csv_path):
        customers_df = spark.read.csv(csv_path, header=True, inferSchema=True)
        print(f"   Loaded {customers_df.count()} customers")
        customers_df.show(5)
    else:
        print(f"   CSV file not found at: {csv_path}")
    
    # Reading from JSON file
    print("\n6. Reading from JSON file:")
    json_path = os.path.join(data_dir, "clickstream.json")
    
    if os.path.exists(json_path):
        clickstream_df = spark.read.json(json_path)
        print(f"   Loaded {clickstream_df.count()} events")
        clickstream_df.show(5, truncate=False)
    else:
        print(f"   JSON file not found at: {json_path}")
    
    # Basic transformations preview
    print("\n7. Basic transformations preview:")
    print("   Filtering:")
    df_tuples.filter(col("age") > 28).show()
    
    print("   Adding computed column:")
    df_tuples.withColumn(
        "salary_category",
        when(col("salary") >= 60000, "High")
        .when(col("salary") >= 50000, "Medium")
        .otherwise("Low")
    ).show()
    
    spark.stop()
    print("\nDataFrame creation examples completed!")


if __name__ == "__main__":
    main()
