"""
Lesson 3.2.1: Explicit Schemas and Type Safety

This script demonstrates:
- Defining explicit schemas
- Schema validation
- Schema evolution handling
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType,
    BooleanType, TimestampType, ArrayType
)
from pyspark.sql.functions import col, current_timestamp


def main():
    spark = SparkSession.builder \
        .appName("Schema Management") \
        .master("local[*]") \
        .getOrCreate()
    
    print("=" * 60)
    print("Schema Management")
    print("=" * 60)
    
    # Example 1: Define explicit schema
    print("\n1. Defining explicit schema:")
    customer_schema = StructType([
        StructField("customer_id", StringType(), False),
        StructField("name", StringType(), False),
        StructField("email", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("is_active", BooleanType(), True),
        StructField("tags", ArrayType(StringType()), True)
    ])
    
    print("   Schema definition:")
    for field in customer_schema.fields:
        print(f"     - {field.name}: {field.dataType.simpleString()} (nullable={field.nullable})")
    
    # Example 2: Create DataFrame with explicit schema
    print("\n2. Creating DataFrame with explicit schema:")
    data = [
        ("C001", "Alice", "alice@email.com", 25, True, ["Premium", "VIP"]),
        ("C002", "Bob", "bob@email.com", 30, True, ["Standard"]),
        ("C003", "Charlie", None, 35, False, []),
    ]
    
    df = spark.createDataFrame(data, customer_schema)
    df.show()
    
    print("   Inferred schema:")
    df.printSchema()
    
    # Example 3: Schema validation
    print("\n3. Schema validation:")
    expected_fields = {"customer_id", "name", "email", "age", "is_active", "tags"}
    actual_fields = set(df.columns)
    
    if expected_fields == actual_fields:
        print("   ✓ Schema validation passed!")
    else:
        missing = expected_fields - actual_fields
        extra = actual_fields - expected_fields
        if missing:
            print(f"   ✗ Missing fields: {missing}")
        if extra:
            print(f"   ✗ Extra fields: {extra}")
    
    # Example 4: Handling schema evolution
    print("\n4. Schema evolution simulation:")
    new_data = [
        ("C004", "Diana", "diana@email.com", 28, True, ["New"]),
        ("C005", "Edward", "edward@email.com", 32, True, ["Standard"]),
    ]
    
    # Add new column to schema
    evolved_schema = StructType([
        StructField("customer_id", StringType(), False),
        StructField("name", StringType(), False),
        StructField("email", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("is_active", BooleanType(), True),
        StructField("tags", ArrayType(StringType()), True),
        StructField("created_at", TimestampType(), True)  # New column
    ])
    
    # Add timestamp to new data
    new_data_with_timestamp = [
        ("C004", "Diana", "diana@email.com", 28, True, ["New"], "2024-01-01"),
        ("C005", "Edward", "edward@email.com", 32, True, ["Standard"], "2024-01-02"),
    ]
    
    df_new = spark.createDataFrame(new_data_with_timestamp, evolved_schema)
    print("   New DataFrame with evolved schema:")
    df_new.printSchema()
    df_new.show()
    
    # Example 5: Common schema patterns
    print("\n5. Common schema patterns:")
    print("""
    Order Schema:
    - order_id: StringType (primary key)
    - customer_id: StringType (foreign key)
    - amount: DoubleType (monetary)
    - quantity: IntegerType (count)
    - order_date: TimestampType (temporal)
    - status: StringType (categorical)
    
    Event Schema:
    - event_id: StringType (primary key)
    - event_type: StringType (categorical)
    - timestamp: TimestampType (temporal)
    - payload: MapType or StructType (semi-structured)
    """)
    
    spark.stop()
    print("\nSchema management demo completed!")


if __name__ == "__main__":
    main()
