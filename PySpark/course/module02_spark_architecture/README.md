# Module 2: Spark Architecture and Execution Model

**Goal:** Understand how Spark executes work so you can write jobs that scale predictably.

## Learning Outcomes

By the end of this module, you will be able to:
- Describe the responsibilities of the Spark driver and executors
- Explain how Spark converts transformations into a physical execution plan
- Distinguish narrow and wide transformations
- Create and configure SparkSession properly
- Understand jobs, stages, tasks, and DAGs

---

## Chapter 2.1: Spark Architecture

### Lesson 2.1.1: Driver, Executors, Cluster Manager

#### Spark Application Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Spark Application                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   Driver    │    │  Cluster    │    │  Executors  │    │
│  │   Process   │◄──►│  Manager    │◄──►│  (Workers)  │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│        │                                      │            │
│        │                                      │            │
│        ▼                                      ▼            │
│  ┌─────────────┐                      ┌─────────────┐     │
│  │SparkContext │                      │    Tasks    │     │
│  │  (Entry    │                      │  (Executed  │     │
│  │   Point)   │                      │  on data)   │     │
│  └─────────────┘                      └─────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Driver Process

The **Driver** is the central coordinator of a Spark application:

- **Responsibilities:**
  - Creates the SparkContext (entry point)
  - Analyzes, optimizes, and schedules work
  - Negotiates resources with the cluster manager
  - Collects results from executors
  - Manages the DAG (Directed Acyclic Graph) of stages

- **What runs on the Driver:**
  - User's main program
  - SparkSession
  - All transformations and actions code
  - Task scheduling logic

- **Common issues:**
  - Driver OOM (Out of Memory) when collecting too much data
  - Single point of failure in non-HA setups

#### Executor Processes

**Executors** are worker processes that execute tasks:

- **Responsibilities:**
  - Execute tasks assigned by the driver
  - Store data partitions in memory or on disk
  - Report status and results back to driver
  - Cache intermediate results if instructed

- **Key properties:**
  - One executor per JVM
  - Multiple tasks can run concurrently (based on cores)
  - Memory is divided between execution and storage

#### Cluster Managers

Cluster managers allocate resources across the cluster:

| Manager | Description | When to Use |
|---------|-------------|-------------|
| **Local** | Single JVM, no cluster | Development, testing |
| **Standalone** | Simple Spark cluster | Small clusters |
| **YARN** | Hadoop resource manager | Hadoop ecosystems |
| **Kubernetes** | Container orchestration | Cloud-native deployments |
| **Mesos** | General-purpose cluster manager | Mixed workloads |
| **Databricks** | Managed Spark platform | Production, managed service |

---

### Lesson 2.1.2: Jobs, Stages, Tasks, and DAGs

#### Execution Hierarchy

```
Application
    │
    └──► Job (triggered by an action)
            │
            └──► Stage (created by shuffle boundaries)
                    │
                    └──► Task (one unit of work per partition)
```

#### Directed Acyclic Graph (DAG)

Spark builds a DAG of transformations before execution:

```python
# This code creates a DAG
df = spark.read.csv("data.csv")           # Source
df = df.filter(col("age") > 25)          # Transformation
df = df.groupBy("department").count()    # Transformation
df.show()                                # Action - triggers execution
```

#### Jobs

- Created when an **action** is called
- Examples of actions: `show()`, `count()`, `collect()`, `write()`
- Each action creates one job

#### Stages

- Created when a **shuffle** occurs
- Shuffle = data movement across partitions
- Wide transformations cause shuffles:
  - `groupBy()`
  - `join()`
  - `orderBy()`
  - `distinct()`

#### Tasks

- One task per partition per stage
- Tasks are the smallest unit of execution
- All tasks in a stage perform the same operation on different data

#### Narrow vs Wide Transformations

| Type | Description | Examples | Shuffle? |
|------|-------------|----------|----------|
| **Narrow** | Each input partition contributes to at most one output partition | `filter`, `select`, `map`, `withColumn` | No |
| **Wide** | Each input partition can contribute to many output partitions | `groupBy`, `join`, `orderBy`, `repartition` | Yes |

---

## Chapter 2.2: SparkSession

### Lesson 2.2.1: Building and Configuring SparkSession

#### SparkSession Configuration

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("My Application") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()
```

#### Common Configuration Options

| Config Key | Description | Default |
|------------|-------------|---------|
| `spark.sql.shuffle.partitions` | Number of partitions after shuffle | 200 |
| `spark.driver.memory` | Driver JVM memory | 1g |
| `spark.executor.memory` | Executor JVM memory | 1g |
| `spark.sql.adaptive.enabled` | Enable adaptive query execution | false (3.x) |
| `spark.sql.autoBroadcastJoinThreshold` | Max size for broadcast joins | 10MB |

#### SparkSession vs SparkContext

- **SparkSession**: Entry point for DataFrame/Dataset API (Spark 2.0+)
- **SparkContext**: Entry point for RDD API (still available)
- SparkSession wraps SparkContext

```python
# Access SparkContext from SparkSession
sc = spark.sparkContext

# Access SQLContext
sql_context = spark._wrapped
```

---

## Hands-On Exercises

### Exercise 1: Architecture Exploration

Create a script that:
1. Creates a SparkSession with custom configuration
2. Reads a dataset
3. Performs filter, join, and aggregation operations
4. Uses `explain()` to examine the execution plan
5. Documents the jobs, stages, and tasks created

### Exercise 2: SparkSession Factory

Build a reusable SparkSession factory that:
1. Accepts app name and environment (dev/test/prod)
2. Applies appropriate configurations for each environment
3. Returns a configured SparkSession

---

## Recommended Project After Module 2

**Diagnostic Spark App**

Create a small diagnostic Spark app that:
1. Reads a dataset
2. Performs filters, joins, and aggregations
3. Documents the generated jobs, stages, tasks, and shuffles
4. Uses Spark UI to visualize execution

See `projects/project02_diagnostic_app/` for the complete implementation.
