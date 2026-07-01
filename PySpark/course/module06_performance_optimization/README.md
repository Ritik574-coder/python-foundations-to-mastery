# Module 6: Performance Optimization and Debugging

**Goal:** Tune Spark jobs and debug failures using evidence instead of guesswork.

## Learning Outcomes

By the end of this module, you will be able to:
- Read logical and physical plans
- Identify expensive shuffles
- Optimize joins and aggregations
- Partition datasets by useful columns
- Use cache and persist intentionally
- Navigate Spark UI for debugging

---

## Chapter 6.1: Performance Optimization

### Lesson 6.1.1: Query Plans, Shuffle, and Join Optimization

#### Catalyst Optimizer

Spark's Catalyst optimizer transforms your query for better performance:

1. **Analysis**: Resolve references, verify types
2. **Logical Optimization**: Apply rule-based optimizations
3. **Physical Planning**: Generate multiple physical plans
4. **Code Generation**: Generate optimized bytecode

#### Reading Execution Plans

```python
# See the plan
df.explain()

# See the analyzed plan
df.explain(True)

# Plan types:
# - Parsed Logical Plan
# - Analyzed Logical Plan
# - Optimized Logical Plan
# - Physical Plan
```

#### Predicate Pushdown

```python
# Bad: Filter after select
df.select("*").filter(col("age") > 25)

# Good: Filter pushed down
df.filter(col("age") > 25).select("name", "age")
```

#### Projection Pruning

```python
# Bad: Select all columns
df = spark.read.parquet("data.parquet")
df.filter(col("age") > 25).show()

# Good: Select only needed columns
df = spark.read.parquet("data.parquet")
df.select("name", "age").filter(col("age") > 25).show()
```

---

## Chapter 6.2: Partitioning and Bucketing

### Lesson 6.2.1: Data Layout for Scalable Queries

#### Directory Partitioning

```python
# Write partitioned by date
df.write \
    .partitionBy("year", "month", "day") \
    .parquet("output/events")

# Resulting directory structure:
# output/events/
#   year=2024/
#     month=01/
#       day=01/
#         part-00000.parquet
#       day=02/
#         part-00000.parquet
```

#### Partition Pruning

```python
# Only reads relevant partitions
df = spark.read.parquet("output/events")
df.filter(col("year") == 2024).filter(col("month") == 1).show()
```

#### Bucketing

```python
# Bucket by column for efficient joins
df.write \
    .bucketBy(16, "customer_id") \
    .sortBy("customer_id") \
    .saveAsTable("bucketed_orders")
```

---

## Chapter 6.3: Caching and Persistence

### Lesson 6.3.1: Reuse Intermediate Results Safely

#### Cache vs Persist

```python
# Cache (in-memory)
df.cache()  # or df.persist()

# Persist with specific storage level
from pyspark import StorageLevel
df.persist(StorageLevel.MEMORY_AND_DISK)

# Unpersist when done
df.unpersist()
```

#### Storage Levels

| Level | Description |
|-------|-------------|
| MEMORY_ONLY | Store as deserialized objects in JVM heap |
| MEMORY_AND_DISK | Spill to disk if not enough memory |
| DISK_ONLY | Store only on disk |
| MEMORY_ONLY_SER | Store as serialized objects |

#### When to Cache

```python
# Good: Reuse same DataFrame multiple times
cleaned_df = raw_df.filter(...).withColumn(...)
cleaned_df.cache()

# Use in multiple aggregations
agg1 = cleaned_df.groupBy("dept").count()
agg2 = cleaned_df.groupBy("region").sum("revenue")

# Bad: Cache then don't reuse
df.cache()
df.show()  # Only used once - cache wasted
```

---

## Chapter 6.4: Spark UI, Error Handling, and Debugging

### Lesson 6.4.1: Debugging Jobs with Spark UI

#### Spark UI Access

```python
# Spark UI is available at:
# http://localhost:4040 (during job execution)
# Or after job completes at history server
```

#### Key Tabs

| Tab | Purpose |
|-----|---------|
| **Jobs** | List of all jobs, duration, status |
| **Stages** | Stages within each job, shuffle read/write |
| **Tasks** | Individual task metrics, skew detection |
| **SQL** | Query plans, execution details |
| **Storage** | Cached DataFrames |

#### Common Issues to Check

1. **Data Skew**: One task taking much longer than others
2. **Shuffle Size**: Large shuffle reads/writes
3. **GC Time**: High garbage collection time
4. **Task Locality**: Tasks not running on data-local nodes

---

### Lesson 6.4.2: Error Handling Patterns

#### Handling Corrupt Records

```python
# Read with corrupt record handling
df = spark.read \
    .option("mode", "PERMISSIVE") \
    .option("columnNameOfCorruptRecord", "_corrupt") \
    .csv("data.csv")

# Separate valid and invalid records
valid_df = df.filter(col("_corrupt").isNull()).drop("_corrupt")
invalid_df = df.filter(col("_corrupt").isNotNull())
```

#### Data Validation

```python
from pyspark.sql.functions import col, when, count, lit

# Validate required fields
def validate_dataframe(df, required_columns):
    validation_results = {}
    for col_name in required_columns:
        null_count = df.filter(col(col_name).isNull()).count()
        validation_results[col_name] = null_count == 0
    return validation_results
```

---

## Hands-On Exercises

### Exercise 1: Query Plan Analysis

Create a script that:
1. Reads a dataset
2. Applies multiple transformations
3. Shows the execution plan at each step
4. Identifies shuffle boundaries

### Exercise 2: Performance Tuning

Build a script that:
1. Creates a slow, unoptimized query
2. Optimizes it step by step
3. Measures performance before and after
4. Documents the improvements

---

## Recommended Project After Module 6

**Optimization Challenge**

Take a deliberately slow Spark job and optimize it using:
1. Partitioning
2. Projection pruning
3. Broadcast joins
4. Caching where appropriate
5. Spark UI evidence

See `projects/project06_optimization/` for the complete implementation.
