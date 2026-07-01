# Module 8: Delta Lake and Incremental Processing

**Goal:** Use Delta Lake and incremental design patterns for reliable lakehouse pipelines.

## Learning Outcomes

By the end of this module, you will be able to:
- Explain why Delta Lake is useful
- Create and query Delta tables
- Use transaction logs and time travel
- Build incremental batch pipelines
- Apply upserts with Delta merge

---

## Chapter 8.1: Delta Lake

### Lesson 8.1.1: Delta Tables and ACID Transactions

#### Why Delta Lake?

| Problem | Traditional Parquet | Delta Lake |
|---------|---------------------|------------|
| ACID transactions | No | Yes |
| Schema enforcement | No | Yes |
| Time travel | No | Yes |
| Concurrent writes | Risk of corruption | Safe |
| Small files | Manual compaction | Auto-compaction |

#### Delta Table Operations

```python
# Write Delta table
df.write.format("delta").mode("overwrite").save("/delta/events")

# Read Delta table
delta_df = spark.read.format("delta").load("/delta/events")

# Time travel
delta_df = spark.read.format("delta").load("/delta/events").option("versionAsOf", 0)

# Check table history
delta_df = DeltaTable.forPath(spark, "/delta/events")
delta_df.history()
```

#### ACID Transactions

```python
# Delta ensures:
# - Atomicity: Write either fully succeeds or fails
# - Consistency: Schema is enforced
# - Isolation: Concurrent reads are safe
# - Durability: Data is persisted

# Example: Safe concurrent writes
df1.write.format("delta").mode("append").save("/delta/events")
df2.write.format("delta").mode("append").save("/delta/events")
```

---

## Chapter 8.2: Incremental Processing

### Lesson 8.2.1: Appends, Upserts, and Change Processing

#### Incremental Load Pattern

```python
def incremental_load(spark, source_path, delta_path, watermark_column):
    # Read new data (using high-water mark)
    last_processed = get_last_processed_date(delta_path)
    
    new_data = spark.read \
        .filter(col(watermark_column) > last_processed) \
        .load(source_path)
    
    # Write to Delta
    new_data.write \
        .format("delta") \
        .mode("append") \
        .save(delta_path)
    
    # Update watermark
    update_last_processed_date(delta_path, new_data.agg(max(watermark_column)).collect()[0][0])
```

#### Delta Merge (Upsert)

```python
from delta.tables import DeltaTable

# Load target table
target = DeltaTable.forPath(spark, "/delta/customers")

# New/updated data
source = spark.read.format("delta").load("/delta/staging/customers")

# Merge (upsert)
target.alias("target") \
    .merge(
        source.alias("source"),
        "target.customer_id = source.customer_id"
    ) \
    .whenMatchedUpdateAll() \
    .whenNotMatchedInsertAll() \
    .execute()
```

#### Handling Late-Arriving Data

```python
def process_with_late_data(spark, source_path, delta_path):
    # Read all new data
    new_data = spark.read.load(source_path)
    
    # Use Delta merge to handle late arrivals
    target = DeltaTable.forPath(spark, delta_path)
    
    target.alias("target").merge(
        new_data.alias("source"),
        "target.order_id = source.order_id"
    ).whenMatchedUpdateAll() \
     .whenNotMatchedInsertAll() \
     .execute()
```

---

## Hands-On Exercises

### Exercise 1: Delta Lake Basics

Create a script that:
1. Writes a DataFrame as Delta table
2. Appends new data
3. Uses time travel to query previous versions
4. Examines table history

### Exercise 2: Incremental Pipeline

Build an incremental pipeline that:
1. Processes only new files from an input folder
2. Tracks last processed timestamp
3. Uses Delta merge for customer updates
4. Handles late-arriving orders

---

## Recommended Project After Module 8

**Incremental Delta Lake Pipeline**

Build an incremental Delta Lake pipeline that:
1. Ingests daily orders
2. Handles updates with merge
3. Supports time travel
4. Produces refreshed gold metrics

See `projects/project08_delta_incremental/` for the complete implementation.
