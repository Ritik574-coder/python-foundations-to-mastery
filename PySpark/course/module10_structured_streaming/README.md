# Module 10: Structured Streaming and Kafka

**Goal:** Build streaming pipelines that ingest, transform, validate, and publish data continuously.

## Learning Outcomes

By the end of this module, you will be able to:
- Explain Spark Structured Streaming
- Read from streaming sources
- Write streaming output with checkpoints
- Connect PySpark to Kafka
- Handle offsets and checkpointing

---

## Chapter 10.1: Structured Streaming

### Lesson 10.1.1: Streaming DataFrames

#### Streaming Concepts

```
┌─────────────────────────────────────────────────────────────┐
│                  Structured Streaming                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   Source    │───►│  Processing │───►│   Sink      │    │
│  │             │    │             │    │             │    │
│  │ - Files     │    │ - Filter    │    │ - Console   │    │
│  │ - Kafka     │    │ - Aggregate │    │ - Parquet   │    │
│  │ - Socket    │    │ - Join      │    │ - Delta     │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Micro-Batch Processing

```python
# Read streaming data
streaming_df = spark.readStream \
    .format("csv") \
    .option("header", "true") \
    .schema(schema) \
    .load("/input/events")

# Write streaming output
query = streaming_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .start()

query.awaitTermination()
```

#### Output Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Append** | Only new rows | Non-aggregation queries |
| **Complete** | Entire result table | Aggregation queries |
| **Update** | Only changed rows | Aggregation with state |

#### Watermarking

```python
# Handle late data with watermark
watermarked_df = streaming_df \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        window("timestamp", "5 minutes"),
        "user_id"
    ).count()
```

#### Checkpointing

```python
# Enable checkpointing for fault tolerance
query = streaming_df.writeStream \
    .format("delta") \
    .option("checkpointLocation", "/checkpoints/events") \
    .start("/output/events")
```

---

## Chapter 10.2: Kafka Integration

### Lesson 10.2.1: Reading and Writing Kafka Streams

#### Reading from Kafka

```python
# Read from Kafka topic
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "events") \
    .option("startingOffsets", "earliest") \
    .load()

# Parse JSON payload
from pyspark.sql.functions import from_json, col

schema = StructType([
    StructField("event_id", StringType()),
    StructField("user_id", StringType()),
    StructField("event_type", StringType()),
    StructField("timestamp", TimestampType())
])

parsed_df = kafka_df \
    .select(
        col("key").cast("string").alias("key"),
        from_json(col("value").cast("string"), schema).alias("data")
    ) \
    .select("key", "data.*")
```

#### Writing to Kafka

```python
# Write to Kafka topic
processed_df.select(
    col("user_id").cast("string").alias("key"),
    to_json(struct("*")).alias("value")
).writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "processed_events") \
    .option("checkpointLocation", "/checkpoints/output") \
    .start()
```

#### Offset Management

```python
# Earliest: Start from beginning
.option("startingOffsets", "earliest")

# Latest: Start from latest
.option("startingOffsets", "latest")

# Specific offset
.option("startingOffsets", '{"topic": {"0": 100, "1": 200}}')
```

---

## Hands-On Exercises

### Exercise 1: File Streaming

Create a streaming pipeline that:
1. Reads CSV files from a directory
2. Aggregates events by time window
3. Writes results to console and Parquet
4. Handles checkpointing for restart

### Exercise 2: Kafka Pipeline

Build a Kafka streaming pipeline that:
1. Reads JSON events from Kafka
2. Parses and validates payloads
3. Aggregates events by window
4. Writes enriched events back to Kafka or Delta

---

## Recommended Project After Module 10

**Streaming Clickstream Pipeline**

Build a streaming clickstream pipeline that:
1. Reads Kafka events
2. Validates payloads
3. Computes session-level metrics
4. Writes clean events to Delta
5. Writes invalid records to quarantine storage

See `projects/project10_streaming_kafka/` for the complete implementation.
