# Module 3: DataFrames, Schemas, and File Formats

**Goal:** Master Spark DataFrames, schema management, and the common file formats used in data lakes.

## Learning Outcomes

By the end of this module, you will be able to:
- Use common DataFrame APIs fluently
- Define explicit schemas using Spark SQL types
- Read and write CSV, JSON, Parquet, and Avro
- Handle schema evolution safely
- Avoid small file problems

---

## Chapter 3.1: DataFrames

### Lesson 3.1.1: DataFrame Operations

#### Core DataFrame Operations

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, upper, trim

spark = SparkSession.builder.getOrCreate()

# Read data
df = spark.read.csv("data.csv", header=True, inferSchema=True)

# Select columns
df.select("name", "age").show()

# Add/modify columns
df = df.withColumn("name_upper", upper(col("name")))
df = df.withColumn("age_group", 
    when(col("age") < 30, "Young")
    .when(col("age") < 50, "Middle")
    .otherwise("Senior")
)

# Drop columns
df = df.drop("name_upper")

# Rename columns
df = df.withColumnRenamed("age", "customer_age")

# Filter rows
df.filter(col("age") > 25).show()
df.where("age > 25 AND name LIKE 'A%'").show()

# Sort
df.orderBy(col("age").desc()).show()

# Deduplicate
df.dropDuplicates(["name"]).show()
```

#### Working with Nested Data

```python
# Create DataFrame with nested structure
data = [
    ("Alice", {"street": "123 Main St", "city": "NYC"}),
    ("Bob", {"street": "456 Oak Ave", "city": "LA"})
]

schema = StructType([
    StructField("name", StringType(), False),
    StructField("address", StructType([
        StructField("street", StringType(), False),
        StructField("city", StringType(), False)
    ]), False)
])

df = spark.createDataFrame(data, schema)

# Access nested fields
df.select(col("address.city")).show()

# Explode arrays
data_with_arrays = [
    ("Alice", ["Python", "Spark"]),
    ("Bob", ["Java", "Scala"])
]
df_arrays = spark.createDataFrame(data_with_arrays, ["name", "skills"])
df_arrays.select(col("name"), explode(col("skills")).alias("skill")).show()
```

---

## Chapter 3.2: Schema Management

### Lesson 3.2.1: Explicit Schemas and Type Safety

#### Schema Definition

```python
from pyspark.sql.types import (
    StructType, StructField, 
    StringType, IntegerType, DoubleType, 
    BooleanType, TimestampType, ArrayType
)

# Define explicit schema
schema = StructType([
    StructField("order_id", StringType(), False),  # Not nullable
    StructField("customer_id", StringType(), False),
    StructField("amount", DoubleType(), False),
    StructField("quantity", IntegerType(), True),  # Nullable
    StructField("is_priority", BooleanType(), True),
    StructField("tags", ArrayType(StringType()), True)
])

# Use schema when reading
df = spark.read.csv("orders.csv", schema=schema, header=True)
```

#### Type Safety Benefits

| Issue | Inferred Schema | Explicit Schema |
|-------|-----------------|-----------------|
| Numeric as string | May infer incorrectly | Enforces correct type |
| Null handling | May miss nulls | Defines nullability |
| Date formats | May fail parsing | Can specify format |
| Schema changes | Silent failures | Throws errors |

---

## Chapter 3.3: Reading and Writing Data

### Lesson 3.3.1: CSV, JSON, Parquet, and Avro

#### CSV

```python
# Read CSV
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("mode", "PERMISSIVE") \
    .option("columnNameOfCorruptRecord", "_corrupt_record") \
    .csv("data.csv")

# Read modes
# PERMISSIVE: puts corrupt records in a column
# DROPMALFORMED: silently ignores bad records
# FAILFAST: throws exception on bad records

# Write CSV
df.write \
    .option("header", "true") \
    .mode("overwrite") \
    .csv("output/csv")
```

#### JSON

```python
# Read JSON (one record per line)
df = spark.read.json("data.json")

# Read multiline JSON
df = spark.read \
    .option("multiLine", "true") \
    .json("multiline.json")

# Write JSON
df.write.mode("overwrite").json("output/json")
```

#### Parquet

```python
# Read Parquet
df = spark.read.parquet("data.parquet")

# Read specific columns (predicate pushdown)
df = spark.read.parquet("data.parquet").select("col1", "col2")

# Write Parquet
df.write.mode("overwrite").parquet("output/parquet")

# Partitioned Parquet
df.write \
    .partitionBy("year", "month") \
    .mode("overwrite") \
    .parquet("output/partitioned")
```

#### File Format Comparison

| Format | Type | Compression | Schema | Best For |
|--------|------|-------------|--------|----------|
| CSV | Row-based | Optional | In header or separate | Interchange, small data |
| JSON | Row-based | Optional | Embedded | Semi-structured data |
| Parquet | Columnar | Built-in | Embedded | Analytics, data lakes |
| Avro | Row-based | Built-in | Embedded | Streaming, evolution |

---

## Hands-On Exercises

### Exercise 1: Schema Definition

Create a script that:
1. Defines an explicit schema for the customers dataset
2. Reads the CSV with the explicit schema
3. Validates the schema matches expectations
4. Handles any schema mismatches gracefully

### Exercise 2: Format Conversion

Build a pipeline that:
1. Reads raw CSV data
2. Converts to Parquet with partitioning
3. Reads back the Parquet data
4. Validates the conversion

---

## Recommended Project After Module 3

**Raw-to-Curated Pipeline**

Build a file conversion pipeline that:
1. Reads CSV and JSON data with explicit schemas
2. Handles malformed records
3. Writes partitioned Parquet datasets

See `projects/project03_raw_to_curated/` for the complete implementation.
