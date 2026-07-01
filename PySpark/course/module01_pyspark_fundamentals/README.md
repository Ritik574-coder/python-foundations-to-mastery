# Module 1: PySpark Fundamentals

**Goal:** Understand what PySpark is, when to use it, and how to start writing simple distributed data processing jobs.

## Learning Outcomes

By the end of this module, you will be able to:
- Explain the role of PySpark in big data systems
- Compare PySpark with pandas, SQL engines, and distributed databases
- Set up a local PySpark development environment
- Create and inspect DataFrames
- Run basic transformations and actions

---

## Chapter 1.1: Introduction to PySpark

### Lesson 1.1.1: What PySpark Solves

#### Key Concepts

**Distributed Processing**
- Processing data across multiple machines (nodes) in parallel
- Enables handling datasets too large for a single machine
- Automatic fault tolerance and recovery

**Apache Spark vs PySpark**
- Apache Spark: Core engine written in Scala
- PySpark: Python API for Spark
- PySpark runs Python code but leverages Spark's JVM-based execution

**Driver and Executors**
- **Driver**: Central coordinator that creates the SparkContext and schedules work
- **Executors**: Worker processes that execute tasks and store data
- **Cluster Manager**: Allocates resources across the cluster (YARN, Mesos, Kubernetes, Standalone)

**Workload Types**
- **Batch**: Process large volumes of data on a schedule (daily, hourly)
- **Streaming**: Process data in real-time as it arrives
- **ETL**: Extract, Transform, Load - moving and transforming data
- **ELT**: Extract, Load, Transform - loading first, transforming in place
- **Feature Pipelines**: Preparing data for machine learning models

#### When to Use PySpark

| Scenario | Use PySpark? | Why |
|----------|--------------|-----|
| 1GB dataset, complex analysis | Maybe | pandas might be sufficient |
| 100GB+ dataset | Yes | Parallel processing required |
| Daily ETL pipeline | Yes | Scalability and fault tolerance |
| Real-time fraud detection | Yes | Low-latency streaming |
| Quick ad-hoc analysis | No | SQL or pandas faster |
| ML training on massive data | Yes | Distributed feature engineering |

#### Real-World Examples

1. **Clickstream Processing**: Processing billions of web events daily for analytics
2. **Data Lake ETL**: Converting raw CSV exports into optimized Parquet tables
3. **Customer Analytics**: Joining customer, order, and payment data for 360° views

---

### Lesson 1.1.2: Local Development Environment

#### Setup Requirements

```
Python 3.10+
Java 8 or 11
PySpark 3.x or 4.x
```

#### Installation Steps

```bash
# 1. Create virtual environment
python -m venv pyspark-env
source pyspark-env/bin/activate  # Linux/Mac
# pyspark-env\Scripts\activate   # Windows

# 2. Install PySpark
pip install pyspark

# 3. Verify installation
python -c "import pyspark; print(pyspark.__version__)"
```

#### Key Concepts

**Local Mode vs Cluster Mode**
- **Local Mode**: Runs on your machine, single JVM, no cluster needed
- **Cluster Mode**: Distributed across multiple machines

**spark-submit**
- Command-line tool for submitting Spark applications
- Supports various cluster managers

**Notebook vs Script**
- **Notebooks**: Interactive, great for exploration
- **Scripts**: Production-ready, version controllable

---

## Chapter 1.2: First PySpark Application

### Lesson 1.2.1: Creating and Inspecting DataFrames

#### Key Concepts

**DataFrame**
- Distributed collection of data organized into named columns
- Similar to pandas DataFrame but distributed across cluster
- Primary abstraction for working with structured data

**Schema**
- Structure definition of a DataFrame
- Column names, data types, nullability

**Lazy Evaluation**
- Transformations are not executed immediately
- Execution happens only when an action is triggered
- Enables optimization by the query planner

#### Core Operations

```python
# Create DataFrame from list of tuples
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
df = spark.createDataFrame(data, ["name", "age"])

# Show data
df.show()

# Display schema
df.printSchema()

# Count rows
df.count()

# Select columns
df.select("name").show()

# Filter rows
df.filter(df.age > 25).show()
```

---

## Hands-On Exercises

### Exercise 1: Hello PySpark

Create your first PySpark application that:
1. Creates a SparkSession
2. Creates a DataFrame from sample data
3. Displays the schema and first 10 rows
4. Counts total records

### Exercise 2: Customer Analysis

Using the sample customers.csv:
1. Read the CSV file
2. Display the schema
3. Show all Premium customers
4. Count customers by state

---

## Recommended Project After Module 1

**Local Orders Pipeline**

Build a PySpark job that:
1. Reads the orders.csv dataset
2. Filters out cancelled orders
3. Calculates daily revenue
4. Writes output as Parquet files

See `projects/project01_orders_pipeline/` for the complete implementation.
