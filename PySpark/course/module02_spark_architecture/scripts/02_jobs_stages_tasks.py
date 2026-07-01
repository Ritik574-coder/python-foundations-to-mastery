"""
Lesson 2.1.2: Jobs, Stages, Tasks, and DAGs

This script demonstrates:
- How Spark creates jobs, stages, and tasks
- Narrow vs wide transformations
- Using explain() to examine execution plans
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as spark_sum


def main():
    spark = SparkSession.builder \
        .appName("Jobs Stages Tasks") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    
    print("=" * 60)
    print("Jobs, Stages, Tasks, and DAGs")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample data:")
    orders_data = [
        ("ORD001", "C001", "Electronics", 100.00),
        ("ORD002", "C002", "Clothing", 50.00),
        ("ORD003", "C001", "Electronics", 200.00),
        ("ORD004", "C003", "Food", 30.00),
        ("ORD005", "C002", "Electronics", 150.00),
        ("ORD006", "C001", "Clothing", 75.00),
        ("ORD007", "C003", "Electronics", 250.00),
        ("ORD008", "C002", "Food", 25.00),
    ]
    
    orders_df = spark.createDataFrame(orders_data, ["order_id", "customer_id", "category", "amount"])
    
    customers_data = [
        ("C001", "Alice", "Premium"),
        ("C002", "Bob", "Standard"),
        ("C003", "Charlie", "Premium"),
    ]
    
    customers_df = spark.createDataFrame(customers_data, ["customer_id", "name", "segment"])
    
    print(f"   Orders: {orders_df.count()} rows")
    print(f"   Customers: {customers_df.count()} rows")
    
    # Example 1: Narrow Transformation (no shuffle)
    print("\n2. Narrow Transformation - Filter:")
    print("   Code: orders_df.filter(col('amount') > 100)")
    filtered_df = orders_df.filter(col("amount") > 100)
    print("   Execution Plan:")
    filtered_df.explain()
    print("   Result:")
    filtered_df.show()
    
    # Example 2: Wide Transformation (shuffle)
    print("\n3. Wide Transformation - GroupBy:")
    print("   Code: orders_df.groupBy('category').agg(sum('amount'))")
    grouped_df = orders_df.groupBy("category").agg(spark_sum("amount").alias("total_amount"))
    print("   Execution Plan:")
    grouped_df.explain()
    print("   Result:")
    grouped_df.show()
    
    # Example 3: Join (wide transformation)
    print("\n4. Wide Transformation - Join:")
    print("   Code: orders_df.join(customers_df, 'customer_id')")
    joined_df = orders_df.join(customers_df, "customer_id")
    print("   Execution Plan:")
    joined_df.explain()
    print("   Result:")
    joined_df.show()
    
    # Example 4: Multiple operations
    print("\n5. Multiple Operations - DAG:")
    print("   Chain: filter -> join -> groupBy -> agg")
    result_df = orders_df \
        .filter(col("amount") > 50) \
        .join(customers_df, "customer_id") \
        .groupBy("segment", "category") \
        .agg(
            count("order_id").alias("order_count"),
            spark_sum("amount").alias("total_amount")
        )
    
    print("   Execution Plan:")
    result_df.explain()
    print("   Result:")
    result_df.show()
    
    # Understanding the plan
    print("\n6. Understanding Execution Plans:")
    print("""
    The execution plan shows:
    - Scan relations (reading data)
    - Filter operations (narrow)
    - BroadcastExchange (optimization for small tables)
    - HashAggregate (groupBy operation)
    - Exchange (shuffle happens here)
    
    Key observations:
    - Filters are pushed down (predicate pushdown)
    - Small tables can be broadcast to avoid shuffle
    - Shuffles create stage boundaries
    """)
    
    spark.stop()
    print("\nJobs, stages, tasks demo completed!")


if __name__ == "__main__":
    main()
