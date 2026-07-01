# Module 9: Batch Processing in Production

**Goal:** Build reliable scheduled Spark batch jobs for real production workflows.

## Learning Outcomes

By the end of this module, you will be able to:
- Design daily, hourly, and backfill jobs
- Parameterize batch runs
- Implement safe overwrite and append patterns
- Add row-count reconciliation

---

## Chapter 9.1: Batch Job Patterns

### Lesson 9.1.1: Scheduled Batch Pipelines

#### Batch Window Design

```python
def process_batch(spark, execution_date):
    # Read data for the batch window
    start_date = execution_date
    end_date = execution_date + timedelta(days=1)
    
    source_df = spark.read \
        .filter(col("event_date") >= start_date) \
        .filter(col("event_date") < end_date) \
        .load(source_path)
    
    # Process and write
    processed_df = transform(source_df)
    
    processed_df.write \
        .mode("overwrite") \
        .partitionBy("event_date") \
        .parquet(output_path)
```

#### Backfill Pattern

```python
def backfill(spark, start_date, end_date):
    current_date = start_date
    
    while current_date <= end_date:
        print(f"Processing batch for {current_date}")
        process_batch(spark, current_date)
        current_date += timedelta(days=1)
```

#### Partition Overwrite

```python
# Safe overwrite for partitioned data
df.write \
    .mode("overwrite") \
    .option("partitionOverwriteMode", "dynamic") \
    .partitionBy("event_date") \
    .parquet(output_path)
```

#### Row Count Reconciliation

```python
def validate_batch(source_df, output_df, batch_date):
    source_count = source_df.filter(col("event_date") == batch_date).count()
    output_count = output_df.filter(col("event_date") == batch_date).count()
    
    if source_count != output_count:
        raise ValueError(f"Count mismatch: source={source_count}, output={output_count}")
    
    return {"batch_date": batch_date, "count": output_count, "status": "success"}
```

---

## Hands-On Exercises

### Exercise 1: Date-Parameterized Job

Create a batch job that:
1. Accepts execution date as parameter
2. Processes data for that date
3. Writes partitioned output
4. Validates row counts

### Exercise 2: Backfill Script

Build a backfill script that:
1. Processes multiple days of data
2. Handles failures gracefully
3. Logs progress
4. Supports resume from failure

---

## Recommended Project After Module 9

**Production Batch Pipeline**

Create a production-style batch pipeline with:
- Date parameters
- Backfill support
- Partition overwrite
- Validation metrics
- Run logs

See `projects/project09_batch_production/` for the complete implementation.
