# Module 11: Data Quality, Testing, and Project Structure

**Goal:** Make PySpark pipelines testable, maintainable, and trustworthy.

## Learning Outcomes

By the end of this module, you will be able to:
- Define data quality checks for Spark datasets
- Separate valid and invalid records
- Write tests for transformation functions
- Structure PySpark applications cleanly

---

## Chapter 11.1: Data Quality Validation

### Lesson 11.1.1: Validation Rules and Quality Gates

#### Data Quality Dimensions

| Dimension | Description | Check |
|-----------|-------------|-------|
| **Completeness** | Required fields present | No nulls in required columns |
| **Uniqueness** | No duplicate records | Primary key uniqueness |
| **Validity** | Values within expected range | Domain constraints |
| **Referential Integrity** | Foreign keys exist | Join validation |
| **Freshness** | Data is up to date | Timestamp checks |

#### Validation Framework

```python
from pyspark.sql import DataFrame
from dataclasses import dataclass
from typing import List

@dataclass
class ValidationResult:
    check_name: str
    passed: bool
    message: str
    failed_count: int

def validate_completeness(df: DataFrame, column: str) -> ValidationResult:
    null_count = df.filter(col(column).isNull()).count()
    return ValidationResult(
        check_name=f"completeness_{column}",
        passed=null_count == 0,
        message=f"Found {null_count} null values in {column}",
        failed_count=null_count
    )

def validate_uniqueness(df: DataFrame, columns: List[str]) -> ValidationResult:
    total_count = df.count()
    unique_count = df.select(columns).distinct().count()
    return ValidationResult(
        check_name=f"uniqueness_{'_'.join(columns)}",
        passed=total_count == unique_count,
        message=f"Found {total_count - unique_count} duplicate records",
        failed_count=total_count - unique_count
    )
```

#### Quarantine Pattern

```python
def quarantine_invalid_records(df, validation_rules):
    valid_records = df
    invalid_records = []
    
    for rule in validation_rules:
        failed_records = valid_records.filter(~rule.condition)
        if failed_records.count() > 0:
            invalid_records.append(failed_records.withColumn("_validation_error", lit(rule.name)))
            valid_records = valid_records.filter(rule.condition)
    
    return valid_records, invalid_records
```

---

## Chapter 11.2: Testing PySpark Pipelines

### Lesson 11.2.1: Unit and Integration Tests

#### Test Fixtures

```python
import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .master("local[*]") \
        .appName("test") \
        .getOrCreate()

@pytest.fixture
def sample_orders(spark):
    data = [
        ("ORD001", "C001", 100.00),
        ("ORD002", "C002", 200.00),
    ]
    return spark.createDataFrame(data, ["order_id", "customer_id", "amount"])
```

#### Unit Testing Transformations

```python
def test_filter_completed_orders(spark):
    # Arrange
    data = [
        ("ORD001", "Completed"),
        ("ORD002", "Cancelled"),
        ("ORD003", "Completed"),
    ]
    df = spark.createDataFrame(data, ["order_id", "status"])
    
    # Act
    result = filter_completed_orders(df)
    
    # Assert
    assert result.count() == 2
    assert "Cancelled" not in [row.status for row in result.collect()]
```

#### Integration Testing

```python
def test_etl_pipeline(spark, sample_data_path):
    # Arrange
    input_df = spark.read.csv(sample_data_path, header=True)
    
    # Act
    result = run_pipeline(spark, input_df)
    
    # Assert
    assert result.count() > 0
    assert "processed_date" in result.columns
    assert result.filter(col("amount") < 0).count() == 0
```

---

## Chapter 11.3: PySpark Project Structure

### Lesson 11.3.1: Organizing Production Code

#### Recommended Structure

```
pyspark_project/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── io/
│   │   ├── __init__.py
│   │   ├── readers.py
│   │   └── writers.py
│   ├── transformations/
│   │   ├── __init__.py
│   │   ├── orders.py
│   │   └── customers.py
│   └── utils/
│       ├── __init__.py
│       ├── spark.py
│       └── logging.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   └── integration/
├── configs/
│   ├── dev.yaml
│   ├── test.yaml
│   └── prod.yaml
├── jobs/
│   ├── __init__.py
│   └── daily_etl.py
├── setup.py
├── requirements.txt
└── README.md
```

#### SparkSession Factory

```python
# src/utils/spark.py
from pyspark.sql import SparkSession

def get_spark_session(app_name, environment="dev"):
    builder = SparkSession.builder.appName(app_name)
    
    if environment == "dev":
        builder = builder.master("local[*]")
    
    return builder.getOrCreate()
```

#### Configuration Management

```python
# src/config/settings.py
import yaml

def load_config(environment):
    with open(f"configs/{environment}.yaml") as f:
        return yaml.safe_load(f)
```

---

## Hands-On Exercises

### Exercise 1: Data Quality Framework

Create a data quality framework that:
1. Defines validation rules
2. Separates valid and invalid records
3. Writes invalid rows to a quarantine path
4. Logs validation results

### Exercise 2: Test Suite

Build a test suite that:
1. Tests transformation functions
2. Uses pytest fixtures for SparkSession
3. Validates schema and row-level output
4. Includes integration tests

---

## Recommended Project After Module 11

**Production Template**

Convert a notebook-based pipeline into a production-style PySpark project with:
- Reusable modules
- Tests
- Configs
- Logging
- Data quality checks

See `projects/project11_production_template/` for the complete implementation.
