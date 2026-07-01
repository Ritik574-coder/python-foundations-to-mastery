"""
Lesson 1.1.2: Local Development Environment

This script demonstrates:
- SparkSession configuration
- Local mode vs cluster mode
- Basic Spark configuration options
"""

from pyspark.sql import SparkSession


def create_spark_session(app_name="Local Dev Environment"):
    """
    Create a SparkSession with common configuration.
    
    In production, these configurations would come from
    environment variables or config files.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()
    
    return spark


def main():
    print("=" * 60)
    print("Local Development Environment Setup")
    print("=" * 60)
    
    # Create SparkSession
    spark = create_spark_session()
    
    # Display Spark configuration
    print("\n1. Spark Configuration:")
    print(f"   App Name: {spark.sparkContext.appName}")
    print(f"   Master: {spark.sparkContext.master}")
    print(f"   Spark Version: {spark.version}")
    
    # Show all configuration
    print("\n2. Key Configuration Settings:")
    config_keys = [
        "spark.sql.shuffle.partitions",
        "spark.driver.memory",
        "spark.executor.memory",
        "spark.sql.adaptive.enabled"
    ]
    
    for key in config_keys:
        try:
            value = spark.conf.get(key)
            print(f"   {key}: {value}")
        except Exception:
            print(f"   {key}: (not set)")
    
    # Test basic functionality
    print("\n3. Testing basic functionality:")
    test_data = [("Test", 1)]
    test_df = spark.createDataFrame(test_data, ["name", "value"])
    print(f"   Created test DataFrame with {test_df.count()} row(s)")
    
    # Show how to run with spark-submit
    print("\n4. Running with spark-submit:")
    print("""
    # Basic spark-submit command:
    spark-submit your_script.py
    
    # With additional options:
    spark-submit \\
        --master local[*] \\
        --driver-memory 4g \\
        --executor-memory 4g \\
        your_script.py
    
    # With configuration:
    spark-submit \\
        --conf spark.sql.shuffle.partitions=200 \\
        your_script.py
    """)
    
    spark.stop()
    print("\nEnvironment setup test completed!")


if __name__ == "__main__":
    main()
