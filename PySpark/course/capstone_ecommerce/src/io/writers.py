"""
Data writers for the E-Commerce Lakehouse Platform.
"""

from pyspark.sql import DataFrame
from typing import List, Optional


class DataWriter:
    """Base data writer class."""
    
    def write_csv(self, df: DataFrame, path: str, mode: str = "overwrite", 
                  header: bool = True, partition_cols: List[str] = None) -> None:
        """Write DataFrame to CSV."""
        writer = df.write.mode(mode).option("header", str(header).lower())
        
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        
        writer.csv(path)
    
    def write_parquet(self, df: DataFrame, path: str, mode: str = "overwrite",
                      partition_cols: List[str] = None) -> None:
        """Write DataFrame to Parquet."""
        writer = df.write.mode(mode)
        
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        
        writer.parquet(path)
    
    def write_delta(self, df: DataFrame, path: str, mode: str = "overwrite",
                    partition_cols: List[str] = None) -> None:
        """Write DataFrame to Delta."""
        writer = df.write.format("delta").mode(mode)
        
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        
        writer.save(path)
    
    def write_json(self, df: DataFrame, path: str, mode: str = "overwrite") -> None:
        """Write DataFrame to JSON."""
        df.write.mode(mode).json(path)


class DeltaWriter(DataWriter):
    """Delta Lake specific writer with merge support."""
    
    def merge(self, target_path: str, source_df: DataFrame, 
              join_condition: str, update_columns: List[str] = None) -> None:
        """
        Merge (upsert) data into Delta table.
        
        Args:
            target_path: Path to target Delta table
            source_df: Source DataFrame
            join_condition: SQL join condition
            update_columns: Columns to update (None = update all)
        """
        from delta.tables import DeltaTable
        
        target = DeltaTable.forPath(source_df.sparkSession, target_path)
        
        merge_builder = target.alias("target") \
            .merge(source_df.alias("source"), join_condition)
        
        if update_columns:
            # Update specific columns
            update_dict = {col: f"source.{col}" for col in update_columns}
            merge_builder = merge_builder.whenMatchedUpdate(set=update_dict)
        else:
            # Update all columns
            merge_builder = merge_builder.whenMatchedUpdateAll()
        
        merge_builder.whenNotMatchedInsertAll().execute()


class QuarantineWriter(DataWriter):
    """Writer for quarantine/invalid records."""
    
    def write_quarantine(self, df: DataFrame, quarantine_path: str, 
                         source_name: str, error_message: str) -> None:
        """
        Write invalid records to quarantine.
        
        Args:
            df: Invalid records DataFrame
            quarantine_path: Base quarantine path
            source_name: Name of the source
            error_message: Description of the validation error
        """
        from pyspark.sql.functions import current_timestamp, lit
        
        quarantine_df = df \
            .withColumn("_quarantine_timestamp", current_timestamp()) \
            .withColumn("_source_name", lit(source_name)) \
            .withColumn("_error_message", lit(error_message))
        
        quarantine_df.write \
            .mode("append") \
            .partitionBy("_source_name") \
            .parquet(quarantine_path)
