# Module 4: Transformations, Actions, SQL, and Analytics

**Goal:** Build reliable analytical transformations using DataFrame APIs and Spark SQL.

## Learning Outcomes

By the end of this module, you will be able to:
- Distinguish transformations from actions
- Explain lazy evaluation in practice
- Build grouped aggregations
- Register DataFrames as temporary views
- Query data with Spark SQL

---

## Chapter 4.1: Transformations and Actions

### Lesson 4.1.1: Lazy Evaluation in Practice

#### Transformations vs Actions

| Type | Description | Examples | Trigger Execution? |
|------|-------------|----------|-------------------|
| **Transformation** | Creates a new DataFrame from existing one | `filter`, `select`, `withColumn`, `groupBy` | No |
| **Action** | Returns a value or writes data | `show`, `count`, `collect`, `write` | Yes |

#### Lazy Evaluation Benefits

```python
# Transformations are recorded but not executed
df = spark.read.csv("data.csv")
df = df.filter(col("age") > 25)  # Not executed yet
df = df.select("name", "age")    # Not executed yet

# Action triggers execution
df.show()  # Now Spark executes all transformations
```

#### When Spark Executes

```python
# Multiple actions create multiple jobs
df1 = df.filter(col("age") > 25)
df2 = df.groupBy("department").count()

df1.show()  # Job 1
df2.show()  # Job 2

# Use cache() to avoid recomputation
df.cache()  # or df.persist()
df1.show()  # Job 1 - computes and caches
df2.show()  # Job 2 - uses cached data
```

---

## Chapter 4.2: Filtering, Sorting, and Aggregations

### Lesson 4.2.1: Core Analytical Operations

#### Filtering

```python
from pyspark.sql.functions import col, when

# Column expression
df.filter(col("age") > 25)

# SQL expression string
df.where("age > 25 AND name LIKE 'A%'")

# Multiple conditions
df.filter(
    (col("age") > 25) & 
    (col("department") == "Engineering")
)

# IS NULL / IS NOT NULL
df.filter(col("email").isNotNull())

# IN clause
df.filter(col("department").isin("Engineering", "Marketing"))
```

#### Aggregations

```python
from pyspark.sql.functions import count, sum, avg, min, max, countDistinct

# Single aggregation
df.groupBy("department").count()

# Multiple aggregations
df.groupBy("department").agg(
    count("employee_id").alias("employee_count"),
    avg("salary").alias("avg_salary"),
    max("salary").alias("max_salary")
)

# Distinct count
df.select(countDistinct("department").alias("dept_count"))

# Conditional aggregation
df.groupBy("department").agg(
    sum(when(col("gender") == "F", 1).otherwise(0)).alias("female_count"),
    sum(when(col("gender") == "M", 1).otherwise(0)).alias("male_count")
)
```

---

## Chapter 4.3: Spark SQL

### Lesson 4.3.1: SQL Views and Querying

#### Temporary Views

```python
# Create temporary view
df.createOrReplaceTempView("employees")

# Query with SQL
result = spark.sql("""
    SELECT department, COUNT(*) as count, AVG(salary) as avg_salary
    FROM employees
    WHERE age > 25
    GROUP BY department
    HAVING COUNT(*) > 5
    ORDER BY avg_salary DESC
""")
result.show()

# Global temporary view (across sessions)
df.createOrReplaceGlobalTempView("global_employees")
spark.sql("SELECT * FROM global_temp.global_employees").show()
```

#### Mixing SQL and DataFrame API

```python
# SQL result is a DataFrame
sql_result = spark.sql("SELECT * FROM employees WHERE age > 25")

# Apply DataFrame transformations
final_result = sql_result \
    .filter(col("department") == "Engineering") \
    .select("name", "salary")

final_result.show()
```

---

## Hands-On Exercises

### Exercise 1: Lazy Evaluation Demo

Create a script that:
1. Chains multiple transformations
2. Uses `explain()` to show the plan before and after actions
3. Demonstrates caching effect on execution time

### Exercise 2: Analytics Mart

Build an analytics mart that:
1. Reads orders, customers, and products data
2. Calculates daily revenue, top products, customer lifetime value
3. Uses both DataFrame APIs and Spark SQL

---

## Recommended Project After Module 4

**Analytics Mart**

Create a Spark analytics mart that:
1. Reads raw orders, payments, and customers
2. Produces gold tables for daily revenue, top products, and customer lifetime value
3. Uses both DataFrame APIs and Spark SQL

See `projects/project04_analytics_mart/` for the complete implementation.
