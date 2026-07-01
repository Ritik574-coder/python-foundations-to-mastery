# Module 13: PySpark for Machine Learning Workloads

**Goal:** Support scalable AI/ML workflows using PySpark for feature engineering, ML pipelines, and inference.

## Learning Outcomes

By the end of this module, you will be able to:
- Understand where Spark fits in ML systems
- Create batch features from large historical datasets
- Build ML pipelines with Spark MLlib
- Apply models at scale using Spark

---

## Chapter 13.1: PySpark for Machine Learning

### Lesson 13.1.1: Spark MLlib Overview

#### MLlib Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Spark MLlib                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │ Feature     │    │  Algorithms │    │   Pipeline  │    │
│  │ Engineering │    │             │    │             │    │
│  │             │    │ - Linear    │    │ - Stages    │    │
│  │ - Vectorizer│    │ - Tree      │    │ - Fit       │    │
│  │ - Scaler    │    │ - Clustering│    │ - Transform │    │
│  │ - Encoder   │    │ - Recommend │    │ - Save/Load │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### When to Use Spark for ML

| Use Case | Use Spark? | Why |
|----------|------------|-----|
| Feature engineering on large data | Yes | Distributed processing |
| Training on 100M+ rows | Yes | Parallel training |
| Small dataset (<1GB) | No | pandas/sklearn faster |
| Real-time inference | Maybe | Consider latency requirements |
| Batch inference at scale | Yes | Distributed scoring |

#### Basic ML Example

```python
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

# Prepare features
assembler = VectorAssembler(
    inputCols=["age", "income", "credit_score"],
    outputCol="features"
)

# Define model
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Create pipeline
pipeline = Pipeline(stages=[assembler, lr])

# Train
model = pipeline.fit(train_df)

# Predict
predictions = model.transform(test_df)
```

---

## Chapter 13.2: Feature Engineering

### Lesson 13.2.1: Building ML Features with PySpark

#### Feature Types

```python
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, StringIndexer,
    OneHotEncoder, Imputer, Bucketizer
)

# Numeric features
assembler = VectorAssembler(inputCols=["age", "income"], outputCol="numeric_features")
scaler = StandardScaler(inputCol="numeric_features", outputCol="scaled_features")

# Categorical features
indexer = StringIndexer(inputCol="department", outputCol="dept_index")
encoder = OneHotEncoder(inputCol="dept_index", outputCol="dept_encoded")

# Handle missing values
imputer = Imputer(
    inputCols=["age", "income"],
    outputCols=["age_imputed", "income_imputed"]
)
```

#### Feature Engineering Patterns

```python
# Recency, Frequency, Monetary (RFM)
from pyspark.sql.functions import datediff, current_date, count, sum

rfm_df = transactions_df.groupBy("customer_id").agg(
    datediff(current_date(), max("timestamp")).alias("recency"),
    count("transaction_id").alias("frequency"),
    sum("amount").alias("monetary")
)

# Time-based features
from pyspark.sql.functions import hour, dayofweek, month

events_df = events_df \
    .withColumn("hour", hour("timestamp")) \
    .withColumn("day_of_week", dayofweek("timestamp")) \
    .withColumn("month", month("timestamp"))
```

#### Preventing Data Leakage

```python
# Point-in-time correctness
def create_training_labels(events_df, prediction_date, label_window_days):
    # Only use events before prediction date
    features = events_df.filter(col("timestamp") < prediction_date)
    
    # Label is based on future events
    labels = events_df.filter(
        (col("timestamp") >= prediction_date) &
        (col("timestamp") < prediction_date + timedelta(days=label_window_days))
    ).groupBy("user_id").agg(count("*").label("label"))
    
    return features, labels
```

---

## Chapter 13.3: ML Pipelines

### Lesson 13.3.1: Reproducible ML Pipelines

#### Pipeline Components

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Feature engineering stages
stage1 = StringIndexer(inputCol="category", outputCol="categoryIndex")
stage2 = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
stage3 = VectorAssembler(inputCols=["categoryVec", "amount", "quantity"], outputCol="features")
stage4 = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# Model stage
stage5 = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="label")

# Create pipeline
pipeline = Pipeline(stages=[stage1, stage2, stage3, stage4, stage5])

# Fit pipeline
model = pipeline.fit(train_df)

# Transform test data
predictions = model.transform(test_df)

# Evaluate
evaluator = BinaryClassificationEvaluator(labelCol="label")
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc}")

# Save pipeline
model.write().overwrite().save("/models/churn_pipeline")
```

#### Cross-Validation

```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=BinaryClassificationEvaluator(),
    numFolds=3
)

cv_model = crossval.fit(train_df)
```

---

## Chapter 13.4: Model Inference with Spark

### Lesson 13.4.1: Batch and Streaming Inference

#### Batch Inference

```python
# Load saved pipeline
loaded_model = PipelineModel.load("/models/churn_pipeline")

# Score new data
new_customers = spark.read.parquet("/data/new_customers")
predictions = loaded_model.transform(new_customers)

# Save predictions
predictions.select("customer_id", "prediction", "probability") \
    .write.mode("overwrite") \
    .parquet("/predictions/churn_scores")
```

#### pandas UDF Scoring

```python
from pyspark.sql.functions import pandas_udf, struct
import pandas as pd
import pickle

@pandas_udf(returnType=StructType([...]))
def score_with_python_model(pdf: pd.DataFrame) -> pd.Series:
    # Load custom model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Score
    predictions = model.predict(pdf[["feature1", "feature2"]])
    return pd.Series(predictions)
```

---

## Hands-On Exercises

### Exercise 1: Feature Engineering

Create a feature engineering pipeline that:
1. Builds customer features from transaction data
2. Handles categorical and numerical fields
3. Assembles feature vectors
4. Creates time-aware training labels

### Exercise 2: ML Pipeline

Build an ML pipeline that:
1. Combines feature transformers and models
2. Persists pipeline stages
3. Scores a test dataset
4. Exports features for external training

---

## Recommended Project After Module 13

**ML Feature and Inference Pipeline**

Build an ML feature and inference pipeline that:
1. Creates customer features
2. Trains a baseline model
3. Runs batch inference
4. Writes predictions to a governed Delta table

See `projects/project13_ml_feature_store/` for the complete implementation.
