"""
Lesson 1.1.1: What PySpark Solves

This script demonstrates:
- Creating a SparkSession
- Basic DataFrame operations
- Comparing PySpark with pandas concepts
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit


def main():
    # Create a SparkSession (entry point to Spark)
    spark = SparkSession.builder \
        .appName("What PySpark Solves") \
        .master("local[*]") \
        .getOrCreate()
    
    print("=" * 60)
    print("PySpark Fundamentals - What PySpark Solves")
    print("=" * 60)
    
    # Example 1: Create DataFrame from Python data
    print("\n1. Creating DataFrame from Python data:")
    data = [
        ("Alice", 25, "Engineering"),
        ("Bob", 30, "Marketing"),
        ("Charlie", 35, "Engineering"),
        ("Diana", 28, "Sales"),
        ("Edward", 32, "Marketing")
    ]
    
    columns = ["name", "age", "department"]
    df = spark.createDataFrame(data, columns)
    
    # Show the DataFrame
    df.show()
    
    # Example 2: Basic operations
    print("\n2. Basic DataFrame operations:")
    print(f"Total rows: {df.count()}")
    print(f"Number of columns: {len(df.columns)}")
    
    # Print schema
    print("\nSchema:")
    df.printSchema()
    
    # Example 3: Filtering
    print("\n3. Filtering - Engineering department:")
    engineering_df = df.filter(col("department") == "Engineering")
    engineering_df.show()
    
    # Example 4: Selection
    print("\n4. Selecting specific columns:")
    df.select("name", "age").show()
    
    # Example 5: Adding a column
    print("\n5. Adding a computed column:")
    df_with_seniority = df.withColumn(
        "is_senior", 
        col("age") > 30
    )
    df_with_seniority.show()
    
    # Example 6: Aggregation
    print("\n6. Aggregation - Average age by department:")
    df.groupBy("department").avg("age").show()
    
    # Comparison with pandas
    print("\n" + "=" * 60)
    print("PySpark vs pandas Comparison:")
    print("=" * 60)
    print("""
    PySpark DataFrame          pandas DataFrame
    -------------------------- --------------------------
    df.show()                  print(df)
    df.count()                 len(df)
    df.filter(col > 5)         df[df.col > 5]
    df.select('col')           df['col']
    df.groupBy().avg()         df.groupby().mean()
    
    Key Differences:
    - PySpark is distributed (scales to TB+ data)
    - PySpark uses lazy evaluation (optimizes queries)
    - pandas is in-memory (limited by RAM)
    """)
    
    # Stop the SparkSession
    spark.stop()
    print("\nExample completed!")


if __name__ == "__main__":
    main()
