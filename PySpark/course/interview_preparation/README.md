# PySpark Interview Preparation

## Topics to Master

### Spark Architecture
- Driver, executors, tasks, stages, and cluster managers
- Lazy evaluation and DAG execution
- DataFrames vs RDDs
- Transformations vs actions
- Narrow vs wide transformations
- Shuffle behavior

### Data Processing
- Join strategies and broadcast joins
- Partitioning, bucketing, and file sizing
- Caching and persistence tradeoffs
- Spark SQL and Catalyst optimizer
- Schema enforcement and schema evolution

### Delta Lake
- ACID transactions
- Merge/upsert operations
- Time travel
- Incremental processing

### Streaming
- Structured Streaming concepts
- Checkpoints, watermarks, and output modes
- Kafka offsets and streaming failure recovery

### Production
- Data quality validation
- Testing PySpark transformations
- Production deployment and CI/CD
- Feature engineering and model inference with Spark

---

## Interview Practice Plan

### Week 1-2: Foundations
1. Explain Spark architecture (driver, executors, stages, tasks)
2. Describe lazy evaluation and DAG execution
3. Compare narrow vs wide transformations
4. Explain shuffle behavior

### Week 3-4: Data Processing
1. Demonstrate join types and strategies
2. Explain partitioning and bucketing
3. Show caching and persistence usage
4. Walk through Catalyst optimizer

### Week 5-6: Advanced Topics
1. Explain Delta Lake ACID transactions
2. Demonstrate merge/upsert operations
3. Show time travel implementation
4. Explain incremental processing patterns

### Week 7-8: Production
1. Walk through a complete project
2. Explain data quality validation
3. Show testing patterns
4. Discuss deployment strategies

---

## Common PySpark Interview Questions

See the complete list in the main README.md file.

### Top 10 Questions to Practice

1. **What is PySpark, and how does it relate to Apache Spark?**
   - PySpark is the Python API for Apache Spark
   - Spark is written in Scala, PySpark provides Python interface
   - PySpark runs Python code but leverages Spark's JVM execution

2. **What is the difference between a job, stage, and task?**
   - Job: Triggered by an action (show, count, write)
   - Stage: Created by shuffle boundaries
   - Task: One unit of work per partition

3. **What is lazy evaluation in Spark?**
   - Transformations are recorded but not executed immediately
   - Execution happens only when an action is triggered
   - Enables optimization by the query planner

4. **What is the difference between transformations and actions?**
   - Transformations: Create new DataFrames (lazy)
   - Actions: Return values or write data (trigger execution)

5. **What are narrow and wide transformations?**
   - Narrow: Each input partition contributes to one output partition (no shuffle)
   - Wide: Each input partition can contribute to many output partitions (shuffle)

6. **What is a broadcast join?**
   - Joins a small table with a large table by broadcasting the small table to all executors
   - Avoids shuffle of the large table
   - Use when one table is small (<10MB)

7. **How do you handle skewed joins?**
   - Use salting to distribute skewed keys
   - Broadcast small tables
   - Use adaptive query execution
   - Repartition data

8. **What is Delta Lake?**
   - Storage layer that adds ACID transactions to data lakes
   - Provides schema enforcement, time travel, and merge operations
   - Built on top of Parquet

9. **How do you optimize a slow Spark job?**
   - Use projection pruning (select only needed columns)
   - Use predicate pushdown (filter early)
   - Broadcast small tables
   - Cache reused DataFrames
   - Tune shuffle partitions
   - Use adaptive query execution

10. **How do you test PySpark transformations?**
    - Use pytest fixtures for SparkSession
    - Create small deterministic datasets
    - Test transformation functions in isolation
    - Validate schema and row-level output

---

## Code Snippets for Interviews

### Join Types
```python
# Inner join
df1.join(df2, "id", "inner")

# Left join
df1.join(df2, "id", "left")

# Left anti join (find unmatched)
df1.join(df2, "id", "left_anti")
```

### Window Functions
```python
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rank, lag

window = Window.partitionBy("department").orderBy("salary")

df.withColumn("rank", rank().over(window))
df.withColumn("prev_salary", lag("salary", 1).over(window))
```

### Delta Merge
```python
from delta.tables import DeltaTable

target = DeltaTable.forPath(spark, "/delta/customers")
target.alias("target").merge(
    source.alias("source"),
    "target.customer_id = source.customer_id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()
```

### Performance Optimization
```python
# Broadcast join
from pyspark.sql.functions import broadcast
df1.join(broadcast(df2), "id")

# Cache
df.cache()
# or
df.persist(StorageLevel.MEMORY_AND_DISK)

# Adaptive query execution
spark.conf.set("spark.sql.adaptive.enabled", "true")
```
