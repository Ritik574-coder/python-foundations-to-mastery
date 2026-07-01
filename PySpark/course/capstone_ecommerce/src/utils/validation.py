"""
Data validation utilities for the E-Commerce Lakehouse Platform.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, when, lit
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a data validation check."""
    check_name: str
    passed: bool
    message: str
    failed_count: int
    total_count: int


class DataValidator:
    """Data validation framework."""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
    
    def validate_completeness(self, df: DataFrame, column: str) -> ValidationResult:
        """
        Validate that a column has no null values.
        
        Args:
            df: DataFrame to validate
            column: Column name to check
            
        Returns:
            ValidationResult
        """
        total_count = df.count()
        null_count = df.filter(col(column).isNull()).count()
        
        result = ValidationResult(
            check_name=f"completeness_{column}",
            passed=null_count == 0,
            message=f"Found {null_count} null values in {column}" if null_count > 0 else f"{column} is complete",
            failed_count=null_count,
            total_count=total_count
        )
        
        self.results.append(result)
        return result
    
    def validate_uniqueness(self, df: DataFrame, columns: List[str]) -> ValidationResult:
        """
        Validate that a combination of columns is unique.
        
        Args:
            df: DataFrame to validate
            columns: List of column names to check
            
        Returns:
            ValidationResult
        """
        total_count = df.count()
        unique_count = df.select(columns).distinct().count()
        duplicate_count = total_count - unique_count
        
        result = ValidationResult(
            check_name=f"uniqueness_{'_'.join(columns)}",
            passed=duplicate_count == 0,
            message=f"Found {duplicate_count} duplicate records" if duplicate_count > 0 else f"Columns are unique",
            failed_count=duplicate_count,
            total_count=total_count
        )
        
        self.results.append(result)
        return result
    
    def validate_range(self, df: DataFrame, column: str, min_value: float = None, 
                       max_value: float = None) -> ValidationResult:
        """
        Validate that a column values are within a range.
        
        Args:
            df: DataFrame to validate
            column: Column name to check
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            ValidationResult
        """
        total_count = df.count()
        
        conditions = []
        if min_value is not None:
            conditions.append(col(column) < min_value)
        if max_value is not None:
            conditions.append(col(column) > max_value)
        
        if conditions:
            combined_condition = conditions[0]
            for condition in conditions[1:]:
                combined_condition = combined_condition | condition
            
            out_of_range_count = df.filter(combined_condition).count()
        else:
            out_of_range_count = 0
        
        result = ValidationResult(
            check_name=f"range_{column}",
            passed=out_of_range_count == 0,
            message=f"Found {out_of_range_count} values out of range" if out_of_range_count > 0 else f"Values are in range",
            failed_count=out_of_range_count,
            total_count=total_count
        )
        
        self.results.append(result)
        return result
    
    def validate_referential_integrity(self, df: DataFrame, foreign_key: str, 
                                       reference_df: DataFrame, reference_key: str) -> ValidationResult:
        """
        Validate referential integrity between two DataFrames.
        
        Args:
            df: DataFrame with foreign key
            foreign_key: Foreign key column name
            reference_df: Reference DataFrame
            reference_key: Reference key column name
            
        Returns:
            ValidationResult
        """
        total_count = df.count()
        
        # Find records without matching reference
        orphans = df.join(
            reference_df,
            df[foreign_key] == reference_df[reference_key],
            "left_anti"
        )
        
        orphan_count = orphans.count()
        
        result = ValidationResult(
            check_name=f"referential_integrity_{foreign_key}",
            passed=orphan_count == 0,
            message=f"Found {orphan_count} orphan records" if orphan_count > 0 else f"Referential integrity maintained",
            failed_count=orphan_count,
            total_count=total_count
        )
        
        self.results.append(result)
        return result
    
    def validate_custom(self, df: DataFrame, check_name: str, 
                        condition, error_message: str) -> ValidationResult:
        """
        Validate with a custom condition.
        
        Args:
            df: DataFrame to validate
            check_name: Name of the check
            condition: PySpark column condition
            error_message: Error message if validation fails
            
        Returns:
            ValidationResult
        """
        total_count = df.count()
        failed_count = df.filter(~condition).count()
        
        result = ValidationResult(
            check_name=check_name,
            passed=failed_count == 0,
            message=error_message if failed_count > 0 else f"Custom check passed",
            failed_count=failed_count,
            total_count=total_count
        )
        
        self.results.append(result)
        return result
    
    def get_summary(self) -> Dict:
        """
        Get summary of all validation results.
        
        Returns:
            Dictionary with validation summary
        """
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.passed)
        failed_checks = total_checks - passed_checks
        
        return {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "pass_rate": passed_checks / total_checks if total_checks > 0 else 0,
            "results": self.results
        }
    
    def print_summary(self):
        """Print validation summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("Data Validation Summary")
        print("=" * 60)
        print(f"Total checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed_checks']}")
        print(f"Failed: {summary['failed_checks']}")
        print(f"Pass rate: {summary['pass_rate']:.2%}")
        print("\nDetailed Results:")
        
        for result in self.results:
            status = "✓" if result.passed else "✗"
            print(f"  {status} {result.check_name}: {result.message}")
        
        print("=" * 60)
