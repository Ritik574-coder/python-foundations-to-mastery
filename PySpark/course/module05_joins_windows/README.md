# Module 5: Joins, Windows, and Advanced Data Processing

**Goal:** Handle relational, time-series, and analytical transformations at scale.

## Learning Outcomes

By the end of this module, you will be able to:
- Use inner, left, right, full, semi, and anti joins
- Build window specifications for ranking and analytics
- Deduplicate records using business rules
- Understand UDF performance tradeoffs

---

## Chapter 5.1: Joins

### Lesson 5.1.1: Join Types and Join Strategy

#### Join Types

```python
# Inner join (only matching records)
df1.join(df2, df1.id == df2.id, "inner")

# Left join (all from left, matching from right)
df1.join(df2, df1.id == df2.id, "left")

# Right join (all from right, matching from left)
df1.join(df2, df1.id == df2.id, "right")

# Full outer join (all records from both)
df1.join(df2, df1.id == df2.id, "full")

# Left semi join (rows from left that have match in right)
df1.join(df2, df1.id == df2.id, "left_semi")

# Left anti join (rows from left that have NO match in right)
df1.join(df2, df1.id == df2.id, "left_anti")
```

#### Join Strategies

| Strategy | When Used | Performance |
|----------|-----------|-------------|
| **Broadcast Join** | One table is small (<10MB) | Fast, no shuffle |
| **Sort-Merge Join** | Both tables are large | Requires shuffle |
| **Shuffle Hash Join** | Medium-sized tables | Requires shuffle |

#### Broadcast Join Hint

```python
from pyspark.sql.functions import broadcast

# Force broadcast join
df1.join(broadcast(df2), "id")

# Or via SQL
spark.sql("SELECT /*+ BROADCAST(df2) */ * FROM df1 JOIN df2 ON df1.id = df2.id")
```

---

## Chapter 5.2: Window Functions

### Lesson 5.2.1: Ranking, Deduplication, and Running Metrics

#### Window Specification

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rank, dense_rank, lag, lead, sum

# Define window
window_spec = Window.partitionBy("department").orderBy("salary")

# Ranking
df.withColumn("rank", rank().over(window_spec))
df.withColumn("dense_rank", dense_rank().over(window_spec))
df.withColumn("row_num", row_number().over(window_spec))

# Running total
running_window = Window.partitionBy("department").orderBy("date") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
df.withColumn("running_total", sum("amount").over(running_window))

# Lag/Lead
df.withColumn("prev_salary", lag("salary", 1).over(window_spec))
df.withColumn("next_salary", lead("salary", 1).over(window_spec))
```

#### Deduplication Pattern

```python
from pyspark.sql import Window
from pyspark.sql.functions import row_number

# Deduplicate by keeping latest record
window_dedup = Window.partitionBy("customer_id").orderBy(col("updated_at").desc())

df_deduped = df \
    .withColumn("row_num", row_number().over(window_dedup)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")
```

---

## Chapter 5.3: User Defined Functions

### Lesson 5.3.1: UDFs, pandas UDFs, and Alternatives

#### Python UDF

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

@udf(returnType=StringType())
def categorize_age(age):
    if age < 30:
        return "Young"
    elif age < 50:
        return "Middle"
    return "Senior"

df.withColumn("age_category", categorize_age(col("age")))
```

#### pandas UDF (Vectorized)

```python
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf(returnType=StringType())
def categorize_age_pandas(age: pd.Series) -> pd.Series:
    return age.apply(lambda x: "Young" if x < 30 else "Middle" if x < 50 else "Senior")

df.withColumn("age_category", categorize_age_pandas(col("age")))
```

#### Performance Comparison

| Type | Speed | Optimizable | Use When |
|------|-------|-------------|----------|
| Built-in functions | Fastest | Yes | Always prefer |
| pandas UDF | Fast | Partial | Complex vectorized logic |
| Python UDF | Slowest | No | No built-in alternative |

---

## Hands-On Exercises

### Exercise 1: Join Types

Create a script that:
1. Creates two sample DataFrames
2. Demonstrates all join types
3. Shows the difference between semi and anti joins
4. Uses broadcast join for a small dimension table

### Exercise 2: Window Functions

Build a script that:
1. Creates time-series data
2. Calculates running totals and moving averages
3. Ranks records within partitions
4. Deduplicates using window functions

---

## Recommended Project After Module 5

**Customer 360 Pipeline**

Build a customer 360 pipeline that:
1. Joins multiple source systems
2. Deduplicates customers with window functions
3. Enriches records with dimensions
4. Creates analytical features

See `projects/project05_customer_360/` for the complete implementation.
