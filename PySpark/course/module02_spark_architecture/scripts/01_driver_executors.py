"""
Lesson 2.1.1: Driver, Executors, Cluster Manager

This script demonstrates:
- Understanding Spark application architecture
- Observing driver and executor behavior
- Configuring parallelism
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, spark_partition_id


def main():
    # Create SparkSession
    spark = SparkSession.builder \
        .appName("Driver and Executors") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    
    print("=" * 60)
    print("Driver, Executors, and Cluster Manager")
    print("=" * 60)
    
    # Get SparkContext
    sc = spark.sparkContext
    
    # Display application information
    print("\n1. Application Information:")
    print(f"   App Name: {sc.appName}")
    print(f"   Master: {sc.master}")
    print(f"   Spark Version: {spark.version}")
    print(f"   Application ID: {sc.applicationId}")
    
    # Display configuration
    print("\n2. Key Configuration:")
    print(f"   Shuffle Partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
    print(f"   Driver Memory: {spark.conf.get('spark.driver.memory')}")
    
    # Create a sample DataFrame
    print("\n3. Creating sample DataFrame:")
    data = [(i, f"user_{i}", i * 100) for i in range(100)]
    df = spark.createDataFrame(data, ["id", "name", "value"])
    
    print(f"   Created DataFrame with {df.count()} rows")
    print(f"   Number of partitions: {df.rdd.getNumPartitions()}")
    
    # Show partition distribution
    print("\n4. Partition Distribution:")
    df.withColumn("partition_id", spark_partition_id()) \
      .groupBy("partition_id") \
      .count() \
      .orderBy("partition_id") \
      .show()
    
    # Demonstrate parallelism
    print("\n5. Parallelism Demo:")
    print("   Each partition is processed by a separate task")
    print("   Tasks are distributed across available cores")
    
    # Show task distribution
    print("\n6. RDD Information:")
    print(f"   Number of partitions in RDD: {df.rdd.getNumPartitions()}")
    print(f"   Number of cores available: {sc.defaultParallelism}")
    
    # Demonstrate shuffle partitions effect
    print("\n7. Shuffle Partitions Effect:")
    print("   Default shuffle partitions: 200")
    print("   Current setting: 4")
    print("   Lower = fewer partitions, less overhead, but potentially larger partitions")
    print("   Higher = more parallelism, but more scheduling overhead")
    
    spark.stop()
    print("\nDriver and Executors demo completed!")


if __name__ == "__main__":
    main()
