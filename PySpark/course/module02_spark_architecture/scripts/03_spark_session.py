"""
Lesson 2.2.1: Building and Configuring SparkSession

This script demonstrates:
- Creating SparkSession with various configurations
- Environment-specific configurations
- SparkSession best practices
"""

from pyspark.sql import SparkSession


class SparkSessionFactory:
    """
    A factory class for creating SparkSessions with environment-specific configurations.
    """
    
    # Environment configurations
    CONFIGS = {
        "dev": {
            "spark.sql.shuffle.partitions": "8",
            "spark.driver.memory": "2g",
            "spark.executor.memory": "2g",
            "spark.sql.adaptive.enabled": "true",
            "spark.ui.enabled": "true"
        },
        "test": {
            "spark.sql.shuffle.partitions": "4",
            "spark.driver.memory": "1g",
            "spark.executor.memory": "1g",
            "spark.sql.adaptive.enabled": "false",
            "spark.ui.enabled": "false"
        },
        "prod": {
            "spark.sql.shuffle.partitions": "200",
            "spark.driver.memory": "4g",
            "spark.executor.memory": "8g",
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.autoBroadcastJoinThreshold": "10485760",  # 10MB
            "spark.ui.enabled": "true"
        }
    }
    
    @classmethod
    def create_session(cls, app_name, environment="dev", additional_configs=None):
        """
        Create a SparkSession with environment-specific configurations.
        
        Args:
            app_name: Name of the application
            environment: Environment type (dev/test/prod)
            additional_configs: Additional configuration overrides
            
        Returns:
            Configured SparkSession
        """
        if environment not in cls.CONFIGS:
            raise ValueError(f"Unknown environment: {environment}. Use: {list(cls.CONFIGS.keys())}")
        
        builder = SparkSession.builder \
            .appName(app_name) \
            .master("local[*]")
        
        # Apply environment configs
        for key, value in cls.CONFIGS[environment].items():
            builder = builder.config(key, value)
        
        # Apply additional configs
        if additional_configs:
            for key, value in additional_configs.items():
                builder = builder.config(key, str(value))
        
        return builder.getOrCreate()


def main():
    print("=" * 60)
    print("SparkSession Configuration")
    print("=" * 60)
    
    # Example 1: Development environment
    print("\n1. Creating Development SparkSession:")
    spark_dev = SparkSessionFactory.create_session(
        app_name="Dev App",
        environment="dev"
    )
    print(f"   App Name: {spark_dev.sparkContext.appName}")
    print(f"   Shuffle Partitions: {spark_dev.conf.get('spark.sql.shuffle.partitions')}")
    print(f"   Driver Memory: {spark_dev.conf.get('spark.driver.memory')}")
    spark_dev.stop()
    
    # Example 2: Production environment
    print("\n2. Creating Production SparkSession:")
    spark_prod = SparkSessionFactory.create_session(
        app_name="Production ETL",
        environment="prod"
    )
    print(f"   App Name: {spark_prod.sparkContext.appName}")
    print(f"   Shuffle Partitions: {spark_prod.conf.get('spark.sql.shuffle.partitions')}")
    print(f"   Driver Memory: {spark_prod.conf.get('spark.driver.memory')}")
    print(f"   Adaptive Query Execution: {spark_prod.conf.get('spark.sql.adaptive.enabled')}")
    spark_prod.stop()
    
    # Example 3: Custom configuration
    print("\n3. Creating SparkSession with Custom Configs:")
    custom_configs = {
        "spark.sql.shuffle.partitions": "16",
        "spark.sql.warehouse.dir": "/tmp/warehouse"
    }
    
    spark_custom = SparkSessionFactory.create_session(
        app_name="Custom App",
        environment="dev",
        additional_configs=custom_configs
    )
    print(f"   App Name: {spark_custom.sparkContext.appName}")
    print(f"   Shuffle Partitions: {spark_custom.conf.get('spark.sql.shuffle.partitions')}")
    spark_custom.stop()
    
    # Best practices
    print("\n" + "=" * 60)
    print("SparkSession Best Practices:")
    print("=" * 60)
    print("""
    1. Use getOrCreate() to reuse existing sessions
    2. Configure shuffle partitions based on data size:
       - Small data (<1GB): 8-16 partitions
       - Medium data (1-100GB): 50-200 partitions
       - Large data (>100GB): 200+ partitions
    
    3. Set driver memory appropriately:
       - Don't collect too much data to driver
       - Use take() or show() instead of collect()
    
    4. Enable adaptive query execution (AQE) in production
       - Automatically optimizes shuffle partitions
       - Handles skew better
    
    5. Always stop SparkSession when done
       - Or let it stop when script exits
    """)
    
    print("\nSparkSession configuration demo completed!")


if __name__ == "__main__":
    main()
