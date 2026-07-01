"""
Lesson 3.1.1: DataFrame Operations

This script demonstrates:
- Common DataFrame APIs
- Column operations
- Nested data handling
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, upper, lower, trim, length,
    concat, lit, explode, arrays_zip
)
from pyspark.sql.types import StructType, StructField, StringType, ArrayType


def main():
    spark = SparkSession.builder \
        .appName("DataFrame Operations") \
        .master("local[*]") \
        .getOrCreate()
    
    print("=" * 60)
    print("DataFrame Operations")
    print("=" * 60)
    
    # Sample data
    data = [
        ("Alice", 25, "Engineering", ["Python", "Spark"]),
        ("Bob", 30, "Marketing", ["SQL", "Tableau"]),
        ("Charlie", 35, "Engineering", ["Java", "Scala"]),
        ("Diana", 28, "Sales", ["CRM", "Excel"]),
        ("Edward", 32, "Marketing", ["Analytics", "Python"])
    ]
    
    columns = ["name", "age", "department", "skills"]
    df = spark.createDataFrame(data, columns)
    
    print("\n1. Original DataFrame:")
    df.show()
    
    # Select operations
    print("\n2. Select specific columns:")
    df.select("name", "age").show()
    
    # Column expressions
    print("\n3. Column expressions:")
    df.select(
        col("name"),
        col("age"),
        upper(col("name")).alias("name_upper"),
        (col("age") * 2).alias("age_doubled")
    ).show()
    
    # Conditional columns
    print("\n4. Conditional column (age group):")
    df.withColumn(
        "age_group",
        when(col("age") < 30, "Young")
        .when(col("age") < 35, "Middle")
        .otherwise("Senior")
    ).show()
    
    # String operations
    print("\n5. String operations:")
    df.select(
        col("name"),
        upper(col("name")).alias("upper_name"),
        length(col("name")).alias("name_length"),
        concat(col("name"), lit(" ("), col("department"), lit(")")).alias("name_dept")
    ).show()
    
    # Rename columns
    print("\n6. Rename columns:")
    df.withColumnRenamed("name", "employee_name").show(3)
    
    # Drop columns
    print("\n7. Drop columns:")
    df.drop("skills").show(3)
    
    # Filter operations
    print("\n8. Filter operations:")
    print("   Age > 30:")
    df.filter(col("age") > 30).show()
    
    print("   Name starts with 'A' or 'B':")
    df.filter(col("name").startswith(("A", "B"))).show()
    
    print("   Department is Engineering:")
    df.where("department = 'Engineering'").show()
    
    # Sorting
    print("\n9. Sorting:")
    df.orderBy(col("age").desc()).show()
    
    # Deduplication
    print("\n10. Deduplication:")
    df_with_dupes = df.union(df)  # Create duplicates
    print(f"    Before dedup: {df_with_dupes.count()} rows")
    df_deduped = df_with_dupes.dropDuplicates(["name"])
    print(f"    After dedup: {df_deduped.count()} rows")
    
    # Working with arrays
    print("\n11. Array operations:")
    df.select(
        col("name"),
        explode(col("skills")).alias("skill")
    ).show()
    
    print("    Array contains check:")
    df.withColumn(
        "knows_python",
        col("skills").contains("Python")
    ).show()
    
    spark.stop()
    print("\nDataFrame operations demo completed!")


if __name__ == "__main__":
    main()
