"""
Exercise 1: Hello PySpark

Complete the tasks below to create your first PySpark application.

Tasks:
1. Create a SparkSession with app name "Hello PySpark"
2. Create a DataFrame with the following data:
   - Columns: id, name, score
   - Data: [(1, "Alice", 95), (2, "Bob", 87), (3, "Charlie", 92), (4, "Diana", 88), (5, "Edward", 91)]
3. Display the schema
4. Show all rows
5. Count total records
6. Filter students with score > 90
7. Calculate average score
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg


def main():
    # TODO: Create a SparkSession
    spark = None  # Replace with your code
    
    # TODO: Create DataFrame
    data = []  # Add your data here
    columns = []  # Add column names
    df = None  # Create DataFrame
    
    # TODO: Display schema
    pass
    
    # TODO: Show all rows
    pass
    
    # TODO: Count total records
    pass
    
    # TODO: Filter students with score > 90
    pass
    
    # TODO: Calculate average score
    pass
    
    # Stop SparkSession
    if spark:
        spark.stop()


if __name__ == "__main__":
    main()
