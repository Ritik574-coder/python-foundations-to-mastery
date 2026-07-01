# Module 7: ETL Pipelines and Data Lake Architecture

**Goal:** Design maintainable PySpark pipelines and data lake layers for analytics and ML consumers.

## Learning Outcomes

By the end of this module, you will be able to:
- Design end-to-end ETL and ELT jobs
- Make jobs idempotent and restartable
- Explain medallion architecture
- Design table responsibilities by layer

---

## Chapter 7.1: ETL Pipelines

### Lesson 7.1.1: Pipeline Design

#### ETL vs ELT

| Pattern | Description | When to Use |
|---------|-------------|-------------|
| **ETL** | Extract, Transform, Load | Transform before loading to target |
| **ELT** | Extract, Load, Transform | Load raw, transform in place |

#### Pipeline Components

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Ingestion  │───►│ Transform   │───►│ Validation  │───►│  Publish    │
│             │    │             │    │             │    │             │
│ - Read      │    │ - Clean     │    │ - Quality   │    │ - Write     │
│ - Parse     │    │ - Join      │    │ - Rejects   │    │ - Notify    │
│ - Validate  │    │ - Aggregate │    │ - Metrics   │    │ - Log       │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

#### Idempotency

```python
# Idempotent write - safe to rerun
def write_idempotent(df, path, partition_columns=None):
    writer = df.write.mode("overwrite")
    
    if partition_columns:
        writer = writer.partitionBy(*partition_columns)
    
    writer.parquet(path)
```

#### Operational Metadata

```python
from pyspark.sql.functions import current_timestamp, lit

# Add metadata columns
df_with_meta = df \
    .withColumn("_ingestion_timestamp", current_timestamp()) \
    .withColumn("_source_file", lit("orders.csv")) \
    .withColumn("_batch_id", lit(batch_id))
```

---

## Chapter 7.2: Data Lake and Medallion Architecture

### Lesson 7.2.1: Bronze, Silver, and Gold Layers

#### Medallion Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Lake                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   Bronze    │───►│   Silver    │───►│    Gold     │    │
│  │             │    │             │    │             │    │
│  │ Raw data    │    │ Cleaned     │    │ Aggregated  │    │
│  │ As-is       │    │ Conformed   │    │ Business    │    │
│  │ Immutable   │    │ Deduplicated│    │ Metrics     │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Layer Responsibilities

| Layer | Purpose | Data Quality | Examples |
|-------|---------|--------------|----------|
| **Bronze** | Raw ingestion | Raw, as-is | CSV, JSON, logs |
| **Silver** | Cleaned, conformed | Validated, deduplicated | Clean orders, customers |
| **Gold** | Business aggregates | Production-ready | Daily revenue, KPIs |

#### Bronze Layer

```python
# Bronze: Raw ingestion
def ingest_to_bronze(spark, source_path, bronze_path, table_name):
    df = spark.read.format(source_format).load(source_path)
    
    df_with_metadata = df \
        .withColumn("_ingestion_date", current_date()) \
        .withColumn("_source_file", input_file_name())
    
    df_with_metadata.write \
        .partitionBy("_ingestion_date") \
        .mode("append") \
        .parquet(f"{bronze_path}/{table_name}")
```

#### Silver Layer

```python
# Silver: Clean and conform
def process_to_silver(spark, bronze_path, silver_path, table_name):
    bronze_df = spark.read.parquet(f"{bronze_path}/{table_name}")
    
    # Clean and validate
    silver_df = bronze_df \
        .filter(col("is_valid") == True) \
        .dropDuplicates(["id"]) \
        .withColumn("processed_date", current_date())
    
    silver_df.write \
        .mode("overwrite") \
        .parquet(f"{silver_path}/{table_name}")
```

#### Gold Layer

```python
# Gold: Aggregate for business use
def process_to_gold(spark, silver_path, gold_path):
    orders_df = spark.read.parquet(f"{silver_path}/orders")
    customers_df = spark.read.parquet(f"{silver_path}/customers")
    
    # Create business metrics
    daily_revenue = orders_df \
        .join(customers_df, "customer_id") \
        .groupBy("order_date", "segment") \
        .agg(
            count("order_id").alias("order_count"),
            sum("amount").alias("revenue")
        )
    
    daily_revenue.write \
        .mode("overwrite") \
        .parquet(f"{gold_path}/daily_revenue")
```

---

## Hands-On Exercises

### Exercise 1: ETL Pipeline

Create a pipeline that:
1. Reads raw CSV data (Bronze)
2. Cleans and validates (Silver)
3. Aggregates for reporting (Gold)
4. Is idempotent and restartable

### Exercise 2: Medallion Lake

Build a medallion data lake for an e-commerce domain with:
1. Raw orders, customers, products, payments
2. Cleaned entities with schema validation
3. Gold metrics for revenue and customer analytics

---

## Recommended Project After Module 7

**Medallion Data Lake**

Build a medallion data lake for an e-commerce domain with:
- Raw orders, customers, products, payments, clickstream events
- Curated entities
- Gold metrics

See `projects/project07_medallion_lake/` for the complete implementation.
