# PySpark Learning Roadmap for Data Engineering and AI/ML Workloads

> **Audience:** Data Engineers who want to build production Spark pipelines and support AI/ML platforms.  
> **Level:** Beginner to Advanced  
> **Primary Focus:** Data Engineering, batch processing, streaming, lakehouse architecture, production pipelines  
> **Secondary Focus:** Feature engineering, ML pipelines, and scalable model inference with Spark

---

## Table of Contents

1. [Course Outcomes](#course-outcomes)
2. [Learning Path](#learning-path)
3. [Module 1: PySpark Fundamentals](#module-1-pyspark-fundamentals)
4. [Module 2: Spark Architecture and Execution Model](#module-2-spark-architecture-and-execution-model)
5. [Module 3: DataFrames, Schemas, and File Formats](#module-3-dataframes-schemas-and-file-formats)
6. [Module 4: Transformations, Actions, SQL, and Analytics](#module-4-transformations-actions-sql-and-analytics)
7. [Module 5: Joins, Windows, and Advanced Data Processing](#module-5-joins-windows-and-advanced-data-processing)
8. [Module 6: Performance Optimization and Debugging](#module-6-performance-optimization-and-debugging)
9. [Module 7: ETL Pipelines and Data Lake Architecture](#module-7-etl-pipelines-and-data-lake-architecture)
10. [Module 8: Delta Lake and Incremental Processing](#module-8-delta-lake-and-incremental-processing)
11. [Module 9: Batch Processing in Production](#module-9-batch-processing-in-production)
12. [Module 10: Structured Streaming and Kafka](#module-10-structured-streaming-and-kafka)
13. [Module 11: Data Quality, Testing, and Project Structure](#module-11-data-quality-testing-and-project-structure)
14. [Module 12: Production Deployment and CI/CD](#module-12-production-deployment-and-cicd)
15. [Module 13: PySpark for Machine Learning Workloads](#module-13-pyspark-for-machine-learning-workloads)
16. [Module 14: End-to-End Data Engineering Projects](#module-14-end-to-end-data-engineering-projects)
17. [Final Capstone Projects](#final-capstone-projects)
18. [Interview Preparation](#interview-preparation)
19. [Common PySpark Interview Questions](#common-pyspark-interview-questions)
20. [Recommended Learning Resources](#recommended-learning-resources)

---

## Course Outcomes

By the end of this course, you should be able to:

- Build reliable PySpark batch and streaming pipelines.
- Read, write, clean, join, aggregate, and transform large datasets.
- Design schemas, partitioning strategies, and lakehouse table layouts.
- Debug Spark jobs using logs, query plans, and Spark UI.
- Optimize jobs using partitioning, caching, broadcast joins, file sizing, and shuffle tuning.
- Implement incremental processing with Delta Lake.
- Integrate Spark with Kafka for streaming ingestion.
- Structure PySpark projects for testing, CI/CD, deployment, and monitoring.
- Support AI/ML workloads with feature engineering, ML pipelines, and distributed inference.

---

## Learning Path

```text
Foundations -> Core Data Engineering -> Optimization -> Lakehouse -> Streaming -> Production -> AI/ML Support
```

Suggested time commitment:

- **Beginner:** 8-10 hours per week for 8-10 weeks
- **Intermediate:** 10-15 hours per week for 10-12 weeks
- **Advanced:** 15+ hours per week with production-style projects

Recommended setup:

- Python 3.10+
- PySpark 3.x or 4.x
- Java runtime compatible with your Spark version
- JupyterLab or VS Code
- Docker for local services
- Optional: Databricks Community Edition, AWS EMR, GCP Dataproc, or Azure Synapse

---

## Module 1: PySpark Fundamentals

**Goal:** Understand what PySpark is, when to use it, and how to start writing simple distributed data processing jobs.

### Chapter 1.1: Introduction to PySpark

#### Lesson 1.1.1: What PySpark Solves

**Learning objectives**

- Explain the role of PySpark in big data systems.
- Compare PySpark with pandas, SQL engines, and distributed databases.
- Identify Data Engineering workloads that fit Spark.

**Key concepts**

- Distributed processing
- Apache Spark vs PySpark
- Driver and executors at a high level
- Batch, streaming, ETL, ELT, and feature pipelines

**Hands-on exercises**

- Install PySpark locally.
- Start a PySpark shell or notebook.
- Create a small DataFrame from Python data.
- Run `show`, `count`, and simple `select` operations.

**Real-world Data Engineering examples**

- Processing daily clickstream logs that are too large for pandas.
- Converting raw CSV exports into curated Parquet tables.
- Joining customer, orders, and payment datasets for analytics.

#### Lesson 1.1.2: Local Development Environment

**Learning objectives**

- Configure a repeatable PySpark development environment.
- Understand dependency requirements for Spark jobs.
- Run a PySpark script from the command line.

**Key concepts**

- Python virtual environments
- Java dependency
- `spark-submit`
- Local mode vs cluster mode
- Notebook vs script workflows

**Hands-on exercises**

- Create a virtual environment.
- Install PySpark.
- Write `hello_spark.py`.
- Run a script with `python` and `spark-submit`.

**Real-world Data Engineering examples**

- Building a local prototype before deploying to Databricks or EMR.
- Reproducing a production transformation bug locally with sample data.

### Chapter 1.2: First PySpark Application

#### Lesson 1.2.1: Creating and Inspecting DataFrames

**Learning objectives**

- Create DataFrames from lists, dictionaries, and files.
- Inspect columns, schema, rows, and basic statistics.
- Understand lazy evaluation at a beginner level.

**Key concepts**

- DataFrame
- Column
- Row
- Schema
- `printSchema`, `show`, `describe`, `explain`

**Hands-on exercises**

- Create a customer DataFrame.
- Select columns and filter records.
- Print schema and row counts.
- Save results as Parquet.

**Real-world Data Engineering examples**

- Validating a small customer extract before loading it into a data lake.
- Inspecting raw ingestion data before applying transformations.

**Recommended project after Module 1**

Build a local PySpark job that reads a small orders dataset, filters invalid records, calculates daily revenue, and writes the output as Parquet.

---

## Module 2: Spark Architecture and Execution Model

**Goal:** Understand how Spark executes work so you can write jobs that scale predictably.

### Chapter 2.1: Spark Architecture

#### Lesson 2.1.1: Driver, Executors, Cluster Manager

**Learning objectives**

- Describe the responsibilities of the Spark driver.
- Explain what executors do.
- Understand the role of cluster managers.

**Key concepts**

- Driver process
- Executor processes
- Cluster managers: Standalone, YARN, Kubernetes, Databricks
- Tasks, jobs, and stages
- Application lifecycle

**Hands-on exercises**

- Run a local job and inspect driver logs.
- Change `master` from `local` to `local[*]`.
- Observe how parallelism changes with input partitions.

**Real-world Data Engineering examples**

- Sizing executor memory and cores for a daily warehouse load.
- Understanding why a driver crashes while collecting too much data.

#### Lesson 2.1.2: Jobs, Stages, Tasks, and DAGs

**Learning objectives**

- Explain how Spark converts transformations into a physical execution plan.
- Distinguish narrow and wide transformations.
- Identify when shuffles happen.

**Key concepts**

- Directed acyclic graph
- Job
- Stage
- Task
- Shuffle
- Narrow transformation
- Wide transformation

**Hands-on exercises**

- Use `explain` on filters, joins, and aggregations.
- Compare execution plans before and after a `groupBy`.
- Trigger actions and observe job creation.

**Real-world Data Engineering examples**

- Diagnosing why a simple aggregation creates expensive network traffic.
- Reducing shuffle cost in a revenue reporting pipeline.

### Chapter 2.2: SparkSession

#### Lesson 2.2.1: Building and Configuring SparkSession

**Learning objectives**

- Create a `SparkSession`.
- Configure app name, master, shuffle partitions, and extensions.
- Understand `SparkContext` and SQL context relationship.

**Key concepts**

- `SparkSession.builder`
- Spark configuration
- `getOrCreate`
- Runtime config
- Session catalog

**Hands-on exercises**

- Create a reusable SparkSession factory.
- Set `spark.sql.shuffle.partitions`.
- Register a temporary view.
- Enable Delta Lake extensions if available.

**Real-world Data Engineering examples**

- Standardizing SparkSession configuration across ETL jobs.
- Using environment-specific configs for development, staging, and production.

**Recommended project after Module 2**

Create a small diagnostic Spark app that reads a dataset, performs filters, joins, and aggregations, and documents the generated jobs, stages, tasks, and shuffles.

---

## Module 3: DataFrames, Schemas, and File Formats

**Goal:** Master Spark DataFrames, schema management, and the common file formats used in data lakes.

### Chapter 3.1: DataFrames

#### Lesson 3.1.1: DataFrame Operations

**Learning objectives**

- Use common DataFrame APIs fluently.
- Select, rename, cast, derive, and drop columns.
- Work with nested and semi-structured data.

**Key concepts**

- `select`
- `withColumn`
- `drop`
- `alias`
- Column expressions
- Nested structs and arrays

**Hands-on exercises**

- Standardize column names.
- Cast dates and numeric fields.
- Extract fields from nested JSON.
- Build a curated customers table.

**Real-world Data Engineering examples**

- Cleaning API ingestion data for downstream analytics.
- Flattening nested event records from mobile applications.

### Chapter 3.2: Schema Management

#### Lesson 3.2.1: Explicit Schemas and Type Safety

**Learning objectives**

- Define explicit schemas using Spark SQL types.
- Avoid schema inference problems.
- Handle schema evolution safely.

**Key concepts**

- `StructType`
- `StructField`
- Nullable fields
- Data types
- Schema inference
- Schema drift

**Hands-on exercises**

- Read CSV with inferred schema and explicit schema.
- Identify incorrect inferred types.
- Add nullable and non-nullable fields.
- Validate schema before processing.

**Real-world Data Engineering examples**

- Preventing a pipeline failure when a vendor changes a column type.
- Enforcing contracts for raw, bronze, silver, and gold tables.

### Chapter 3.3: Reading and Writing Data

#### Lesson 3.3.1: CSV, JSON, Parquet, and Avro

**Learning objectives**

- Read and write CSV, JSON, Parquet, and Avro.
- Choose the right format for different Data Engineering workloads.
- Configure read and write options.

**Key concepts**

- CSV headers, delimiters, quotes, corrupt records
- JSON multiline and nested records
- Parquet columnar storage
- Avro row-based serialization
- Compression
- Read modes: permissive, drop malformed, fail fast

**Hands-on exercises**

- Read a raw CSV file with an explicit schema.
- Read nested JSON logs.
- Convert CSV and JSON to Parquet.
- Write Avro records for downstream systems.

**Real-world Data Engineering examples**

- Landing vendor CSV files in bronze storage.
- Storing curated analytics tables as Parquet.
- Publishing Avro messages for schema-aware consumers.

#### Lesson 3.3.2: Write Modes and Data Layout

**Learning objectives**

- Use append, overwrite, ignore, and error modes.
- Understand file output behavior.
- Avoid small file problems.

**Key concepts**

- `mode`
- `coalesce`
- `repartition`
- File sizes
- Output directories
- Atomicity considerations

**Hands-on exercises**

- Write append and overwrite outputs.
- Compare number of output files after repartitioning.
- Create partitioned output by date.

**Real-world Data Engineering examples**

- Writing daily sales partitions to a data lake.
- Controlling output file sizes for faster downstream queries.

**Recommended project after Module 3**

Build a raw-to-curated file conversion pipeline that reads CSV and JSON data with explicit schemas, handles malformed records, and writes partitioned Parquet datasets.

---

## Module 4: Transformations, Actions, SQL, and Analytics

**Goal:** Build reliable analytical transformations using DataFrame APIs and Spark SQL.

### Chapter 4.1: Transformations and Actions

#### Lesson 4.1.1: Lazy Evaluation in Practice

**Learning objectives**

- Distinguish transformations from actions.
- Explain when Spark actually executes code.
- Avoid expensive actions in production jobs.

**Key concepts**

- Lazy evaluation
- Transformations
- Actions
- Lineage
- `count`, `collect`, `take`, `write`

**Hands-on exercises**

- Chain multiple transformations and trigger one action.
- Compare `show`, `take`, and `collect`.
- Use `explain` before running an action.

**Real-world Data Engineering examples**

- Avoiding `collect` on a billion-row table.
- Building transformation logic that executes only at final write time.

### Chapter 4.2: Filtering, Sorting, and Aggregations

#### Lesson 4.2.1: Core Analytical Operations

**Learning objectives**

- Filter records using column expressions.
- Sort data correctly.
- Build grouped aggregations.

**Key concepts**

- `filter` and `where`
- `orderBy`
- `groupBy`
- `agg`
- Built-in functions
- Null handling

**Hands-on exercises**

- Filter orders by date and status.
- Calculate revenue by day and category.
- Sort top products by sales.
- Handle null customer IDs and invalid prices.

**Real-world Data Engineering examples**

- Creating a gold daily revenue table.
- Detecting inactive customers from transaction history.

### Chapter 4.3: Spark SQL

#### Lesson 4.3.1: SQL Views and Querying

**Learning objectives**

- Register DataFrames as temporary views.
- Query data with Spark SQL.
- Mix SQL and DataFrame APIs.

**Key concepts**

- Temporary views
- Global temporary views
- SQL catalog
- SQL expressions
- CTEs

**Hands-on exercises**

- Register orders and customers as views.
- Write SQL aggregations.
- Convert SQL results back to DataFrames.
- Compare SQL and DataFrame execution plans.

**Real-world Data Engineering examples**

- Supporting analysts who prefer SQL while keeping engineering pipelines in PySpark.
- Migrating legacy warehouse SQL into Spark jobs.

**Recommended project after Module 4**

Create a Spark analytics mart that reads raw orders, payments, and customers, then produces gold tables for daily revenue, top products, and customer lifetime value using both DataFrame APIs and Spark SQL.

---

## Module 5: Joins, Windows, and Advanced Data Processing

**Goal:** Handle relational, time-series, and analytical transformations at scale.

### Chapter 5.1: Joins

#### Lesson 5.1.1: Join Types and Join Strategy

**Learning objectives**

- Use inner, left, right, full, semi, and anti joins.
- Understand duplicate keys and join explosion.
- Choose safe join patterns for pipelines.

**Key concepts**

- Equi joins
- Non-equi joins
- Broadcast joins
- Shuffle joins
- Skew
- Semi and anti joins

**Hands-on exercises**

- Join orders to customers and products.
- Use left anti join to find unmatched records.
- Broadcast a small dimension table.
- Detect duplicate keys before joining.

**Real-world Data Engineering examples**

- Enriching event logs with user profile data.
- Finding orphan records during data reconciliation.

### Chapter 5.2: Window Functions

#### Lesson 5.2.1: Ranking, Deduplication, and Running Metrics

**Learning objectives**

- Build window specifications.
- Use ranking and analytic functions.
- Deduplicate records using business rules.

**Key concepts**

- `Window.partitionBy`
- `orderBy`
- `row_number`
- `rank`
- `dense_rank`
- `lag`
- `lead`
- Running totals

**Hands-on exercises**

- Deduplicate customer records by latest update timestamp.
- Calculate running revenue by customer.
- Identify previous and next event timestamps.
- Rank products by category revenue.

**Real-world Data Engineering examples**

- Building slowly changing dimension logic.
- Creating customer behavior features for churn models.

### Chapter 5.3: User Defined Functions

#### Lesson 5.3.1: UDFs, pandas UDFs, and Alternatives

**Learning objectives**

- Create Python UDFs when needed.
- Understand UDF performance tradeoffs.
- Prefer built-in functions where possible.

**Key concepts**

- Python UDF
- pandas UDF
- Serialization cost
- Catalyst optimizer limitations
- Built-in Spark SQL functions

**Hands-on exercises**

- Implement a simple standardization UDF.
- Replace a UDF with built-in functions.
- Benchmark built-in functions vs Python UDFs on sample data.
- Use pandas UDF for vectorized logic.

**Real-world Data Engineering examples**

- Standardizing messy product codes.
- Applying custom text normalization before feature extraction.

**Recommended project after Module 5**

Build a customer 360 pipeline that joins multiple source systems, deduplicates customers with window functions, enriches records with dimensions, and creates analytical features.

---

## Module 6: Performance Optimization and Debugging

**Goal:** Tune Spark jobs and debug failures using evidence instead of guesswork.

### Chapter 6.1: Performance Optimization

#### Lesson 6.1.1: Query Plans, Shuffle, and Join Optimization

**Learning objectives**

- Read logical and physical plans.
- Identify expensive shuffles.
- Optimize joins and aggregations.

**Key concepts**

- Catalyst optimizer
- Tungsten execution engine
- Predicate pushdown
- Projection pruning
- Adaptive query execution
- Broadcast hints

**Hands-on exercises**

- Compare query plans before and after selecting fewer columns.
- Enable adaptive query execution.
- Use broadcast join hints.
- Reduce shuffle partitions for small datasets.

**Real-world Data Engineering examples**

- Tuning a slow daily KPI pipeline.
- Reducing cloud compute cost for recurring ETL jobs.

### Chapter 6.2: Partitioning and Bucketing

#### Lesson 6.2.1: Data Layout for Scalable Queries

**Learning objectives**

- Partition datasets by useful columns.
- Understand when bucketing helps.
- Avoid over-partitioning.

**Key concepts**

- Partition pruning
- Directory partitioning
- Bucketing
- Sort within partitions
- File size management
- Data skew

**Hands-on exercises**

- Write data partitioned by event date.
- Query one date partition and inspect file scans.
- Create a bucketed table if using a catalog-backed environment.
- Compare partition counts and query performance.

**Real-world Data Engineering examples**

- Partitioning clickstream data by event date.
- Designing table layout for customer and order joins.

### Chapter 6.3: Caching and Persistence

#### Lesson 6.3.1: Reuse Intermediate Results Safely

**Learning objectives**

- Use cache and persist intentionally.
- Choose storage levels.
- Unpersist data when no longer needed.

**Key concepts**

- `cache`
- `persist`
- Storage levels
- Memory pressure
- Re-computation
- `unpersist`

**Hands-on exercises**

- Cache a reused DataFrame.
- Compare execution time with and without caching.
- Check cached storage in Spark UI.
- Unpersist intermediate DataFrames.

**Real-world Data Engineering examples**

- Reusing cleaned events for multiple downstream aggregations.
- Avoiding unnecessary caching in one-pass ETL jobs.

### Chapter 6.4: Spark UI, Error Handling, and Debugging

#### Lesson 6.4.1: Debugging Jobs with Spark UI

**Learning objectives**

- Navigate Spark UI.
- Interpret jobs, stages, tasks, SQL, storage, and environment tabs.
- Diagnose common bottlenecks.

**Key concepts**

- Spark UI
- Stage duration
- Task skew
- Shuffle read/write
- Executor logs
- SQL execution details

**Hands-on exercises**

- Run a job and inspect Spark UI.
- Identify a shuffle-heavy operation.
- Find failed tasks and executor errors.
- Compare plans for optimized and unoptimized jobs.

**Real-world Data Engineering examples**

- Diagnosing a slow join due to skewed customer IDs.
- Finding executor out-of-memory causes from logs and UI metrics.

#### Lesson 6.4.2: Error Handling Patterns

**Learning objectives**

- Handle corrupt input records.
- Build validation and quarantine flows.
- Log useful failure context.

**Key concepts**

- Read modes
- Bad record paths
- Try-except boundaries
- Data validation failures
- Dead-letter datasets
- Idempotency

**Hands-on exercises**

- Read malformed CSV and JSON records.
- Separate valid and invalid rows.
- Write invalid rows to a quarantine path.
- Add structured logging around a pipeline.

**Real-world Data Engineering examples**

- Quarantining vendor records that break schema contracts.
- Reprocessing failed partitions after upstream correction.

**Recommended project after Module 6**

Take a deliberately slow and unstable Spark job, then optimize it using partitioning, projection pruning, broadcast joins, caching where appropriate, and Spark UI evidence.

---

## Module 7: ETL Pipelines and Data Lake Architecture

**Goal:** Design maintainable PySpark pipelines and data lake layers for analytics and ML consumers.

### Chapter 7.1: ETL Pipelines

#### Lesson 7.1.1: Pipeline Design

**Learning objectives**

- Design end-to-end ETL and ELT jobs.
- Separate ingestion, transformation, validation, and publishing.
- Make jobs idempotent and restartable.

**Key concepts**

- ETL vs ELT
- Idempotency
- Checkpoints
- Reprocessing
- Source-to-target mapping
- Operational metadata

**Hands-on exercises**

- Build bronze, silver, and gold transformations.
- Add run date parameters.
- Re-run the same job without duplicate output.
- Capture row counts and validation metrics.

**Real-world Data Engineering examples**

- Daily ingestion from CRM exports into an analytics lake.
- Building curated datasets for BI dashboards and ML training.

### Chapter 7.2: Data Lake and Medallion Architecture

#### Lesson 7.2.1: Bronze, Silver, and Gold Layers

**Learning objectives**

- Explain medallion architecture.
- Design table responsibilities by layer.
- Serve analytics and ML use cases from curated layers.

**Key concepts**

- Bronze raw data
- Silver cleaned and conformed data
- Gold business aggregates
- Data contracts
- Lineage
- Governance

**Hands-on exercises**

- Create bronze raw events.
- Clean and deduplicate into silver.
- Aggregate into gold fact tables.
- Document data lineage.

**Real-world Data Engineering examples**

- Supporting BI dashboards from gold tables.
- Supporting ML features from clean silver event history.

**Recommended project after Module 7**

Build a medallion data lake for an e-commerce domain with raw orders, customers, products, payments, clickstream events, curated entities, and gold metrics.

---

## Module 8: Delta Lake and Incremental Processing

**Goal:** Use Delta Lake and incremental design patterns for reliable lakehouse pipelines.

### Chapter 8.1: Delta Lake

#### Lesson 8.1.1: Delta Tables and ACID Transactions

**Learning objectives**

- Explain why Delta Lake is useful.
- Create and query Delta tables.
- Use transaction logs and time travel.

**Key concepts**

- ACID transactions
- Delta transaction log
- Time travel
- Schema enforcement
- Schema evolution
- Table history

**Hands-on exercises**

- Write a DataFrame as Delta.
- Append and overwrite Delta data.
- Query table history.
- Read a previous table version.

**Real-world Data Engineering examples**

- Preventing partial writes in production ETL.
- Recovering from a bad deployment using time travel.

### Chapter 8.2: Incremental Processing

#### Lesson 8.2.1: Appends, Upserts, and Change Processing

**Learning objectives**

- Build incremental batch pipelines.
- Use watermarks or high-water marks.
- Apply upserts with Delta merge.

**Key concepts**

- Incremental loads
- High-water mark
- CDC
- Merge/upsert
- Late-arriving data
- Idempotent writes

**Hands-on exercises**

- Process only new files from an input folder.
- Track last processed timestamp.
- Use Delta merge for customer updates.
- Handle late-arriving orders.

**Real-world Data Engineering examples**

- Loading only new transactions each day.
- Applying customer profile updates from a source system.

**Recommended project after Module 8**

Build an incremental Delta Lake pipeline that ingests daily orders, handles updates with merge, supports time travel, and produces refreshed gold metrics.

---

## Module 9: Batch Processing in Production

**Goal:** Build reliable scheduled Spark batch jobs for real production workflows.

### Chapter 9.1: Batch Job Patterns

#### Lesson 9.1.1: Scheduled Batch Pipelines

**Learning objectives**

- Design daily, hourly, and backfill jobs.
- Parameterize batch runs.
- Implement safe overwrite and append patterns.

**Key concepts**

- Batch windows
- Backfills
- Run parameters
- Partition overwrite
- SLA
- Retry behavior

**Hands-on exercises**

- Build a date-parameterized job.
- Run a one-day batch and a multi-day backfill.
- Write partition-specific output.
- Add row-count reconciliation.

**Real-world Data Engineering examples**

- Backfilling six months of order history.
- Rebuilding a broken daily reporting partition.

**Recommended project after Module 9**

Create a production-style batch pipeline with date parameters, backfill support, partition overwrite, validation metrics, and run logs.

---

## Module 10: Structured Streaming and Kafka

**Goal:** Build streaming pipelines that ingest, transform, validate, and publish data continuously.

### Chapter 10.1: Structured Streaming

#### Lesson 10.1.1: Streaming DataFrames

**Learning objectives**

- Explain Spark Structured Streaming.
- Read from streaming sources.
- Write streaming output with checkpoints.

**Key concepts**

- Micro-batch processing
- Streaming DataFrame
- Trigger
- Output modes
- Checkpointing
- Watermarking

**Hands-on exercises**

- Read streaming files from a directory.
- Aggregate events by time window.
- Write results to a console and Parquet sink.
- Restart a stream from checkpoint.

**Real-world Data Engineering examples**

- Processing near-real-time application events.
- Updating operational dashboards every few minutes.

### Chapter 10.2: Kafka Integration

#### Lesson 10.2.1: Reading and Writing Kafka Streams

**Learning objectives**

- Connect PySpark to Kafka.
- Parse Kafka key and value payloads.
- Handle offsets and checkpointing.

**Key concepts**

- Kafka topic
- Partitions and offsets
- Consumer groups
- JSON payload parsing
- Checkpoint location
- Exactly-once considerations

**Hands-on exercises**

- Read JSON events from Kafka.
- Parse value into structured columns.
- Aggregate events by window.
- Write enriched events back to Kafka or Delta.

**Real-world Data Engineering examples**

- Ingesting clickstream events from Kafka into a lakehouse.
- Building fraud signal features from payment event streams.

**Recommended project after Module 10**

Build a streaming clickstream pipeline that reads Kafka events, validates payloads, computes session-level metrics, writes clean events to Delta, and writes invalid records to quarantine storage.

---

## Module 11: Data Quality, Testing, and Project Structure

**Goal:** Make PySpark pipelines testable, maintainable, and trustworthy.

### Chapter 11.1: Data Quality Validation

#### Lesson 11.1.1: Validation Rules and Quality Gates

**Learning objectives**

- Define data quality checks for Spark datasets.
- Separate valid and invalid records.
- Fail or warn based on severity.

**Key concepts**

- Completeness
- Uniqueness
- Validity
- Referential integrity
- Freshness
- Distribution checks

**Hands-on exercises**

- Validate required fields.
- Detect duplicate primary keys.
- Check order totals are non-negative.
- Validate foreign keys between orders and customers.

**Real-world Data Engineering examples**

- Blocking bad customer data before it reaches BI reports.
- Monitoring feature freshness for ML training datasets.

### Chapter 11.2: Testing PySpark Pipelines

#### Lesson 11.2.1: Unit and Integration Tests

**Learning objectives**

- Write tests for transformation functions.
- Compare DataFrame outputs.
- Use small deterministic datasets.

**Key concepts**

- Test SparkSession
- Pure transformation functions
- Fixtures
- DataFrame equality
- Integration tests
- Golden datasets

**Hands-on exercises**

- Refactor transformations into testable functions.
- Create pytest fixtures for SparkSession.
- Test schema and row-level output.
- Add an integration test for a mini pipeline.

**Real-world Data Engineering examples**

- Preventing regression in revenue calculation logic.
- Testing deduplication rules before deploying to production.

### Chapter 11.3: PySpark Project Structure

#### Lesson 11.3.1: Organizing Production Code

**Learning objectives**

- Structure PySpark applications cleanly.
- Separate configuration, IO, transformations, and orchestration.
- Package jobs for deployment.

**Key concepts**

- `src` layout
- Config files
- Job entry points
- Reusable transformations
- Logging
- Dependency management

**Hands-on exercises**

- Create a project with `src`, `tests`, `jobs`, and `configs`.
- Add a reusable SparkSession factory.
- Add structured logging.
- Package and run a job.

**Real-world Data Engineering examples**

- Maintaining multiple jobs in one data platform repository.
- Sharing common transformation logic across batch and streaming jobs.

**Recommended project after Module 11**

Convert a notebook-based pipeline into a production-style PySpark project with reusable modules, tests, configs, logging, and data quality checks.

---

## Module 12: Production Deployment and CI/CD

**Goal:** Deploy PySpark jobs reliably and automate quality checks before release.

### Chapter 12.1: Production Deployment

#### Lesson 12.1.1: Running Spark Jobs in Real Environments

**Learning objectives**

- Compare deployment targets for Spark.
- Submit jobs with dependencies and configuration.
- Understand operational concerns.

**Key concepts**

- `spark-submit`
- Databricks Jobs
- EMR Steps
- Dataproc jobs
- Kubernetes Spark Operator
- Secrets and environment variables
- Monitoring and alerting

**Hands-on exercises**

- Create a deployable job entry point.
- Pass runtime parameters.
- Package dependencies.
- Simulate development and production config files.

**Real-world Data Engineering examples**

- Deploying a daily Delta Lake merge job.
- Running Spark pipelines as part of an orchestrated workflow.

### Chapter 12.2: CI/CD for PySpark

#### Lesson 12.2.1: Automated Checks and Releases

**Learning objectives**

- Build CI checks for PySpark repositories.
- Run unit tests and linting automatically.
- Promote jobs between environments.

**Key concepts**

- CI pipeline
- CD pipeline
- Linting
- Unit tests
- Integration tests
- Build artifacts
- Environment promotion

**Hands-on exercises**

- Add pytest test execution to CI.
- Add code formatting and linting checks.
- Build a deployable artifact.
- Create a deployment checklist.

**Real-world Data Engineering examples**

- Preventing broken ETL code from reaching production.
- Releasing a new feature table pipeline through staging first.

**Recommended project after Module 12**

Build a CI/CD-ready PySpark repository with tests, linting, environment configs, packaged jobs, and a deployment guide for Databricks, EMR, or Kubernetes.

---

## Module 13: PySpark for Machine Learning Workloads

**Goal:** Support scalable AI/ML workflows using PySpark for feature engineering, ML pipelines, and inference.

### Chapter 13.1: PySpark for Machine Learning

#### Lesson 13.1.1: Spark MLlib Overview

**Learning objectives**

- Understand where Spark fits in ML systems.
- Use Spark for distributed preprocessing and training where appropriate.
- Know when to move from Spark to specialized ML frameworks.

**Key concepts**

- MLlib
- Estimators
- Transformers
- Pipeline API
- Distributed feature processing
- Train/test split

**Hands-on exercises**

- Load a training dataset.
- Build a simple classification or regression model.
- Evaluate model metrics.
- Save and load a Spark ML model.

**Real-world Data Engineering examples**

- Preparing large training datasets for churn prediction.
- Training baseline models on distributed customer features.

### Chapter 13.2: Feature Engineering

#### Lesson 13.2.1: Building ML Features with PySpark

**Learning objectives**

- Create batch features from large historical datasets.
- Handle categorical, numerical, and text fields.
- Prevent data leakage.

**Key concepts**

- Feature tables
- Point-in-time correctness
- Categorical encoding
- VectorAssembler
- StandardScaler
- StringIndexer
- OneHotEncoder
- Text tokenization

**Hands-on exercises**

- Build customer recency, frequency, and monetary features.
- Encode categorical columns.
- Assemble feature vectors.
- Create time-aware training labels.

**Real-world Data Engineering examples**

- Building features for churn, fraud, recommendation, or demand forecasting models.
- Creating reusable feature tables for a feature store.

### Chapter 13.3: ML Pipelines

#### Lesson 13.3.1: Reproducible ML Pipelines

**Learning objectives**

- Combine feature transformers and models into an ML pipeline.
- Persist pipeline stages.
- Prepare datasets for downstream model training systems.

**Key concepts**

- `Pipeline`
- `PipelineModel`
- Transformers
- Estimators
- Evaluators
- Cross-validation basics

**Hands-on exercises**

- Build a pipeline with indexing, encoding, assembling, and model training.
- Save and reload a trained pipeline.
- Score a test dataset.
- Export features for external training.

**Real-world Data Engineering examples**

- Standardizing feature processing between training and batch inference.
- Producing model-ready datasets for data scientists.

### Chapter 13.4: Model Inference with Spark

#### Lesson 13.4.1: Batch and Streaming Inference

**Learning objectives**

- Apply models at scale using Spark.
- Understand inference patterns and limitations.
- Use pandas UDFs for custom model scoring.

**Key concepts**

- Batch inference
- Streaming inference
- Model loading
- pandas UDF scoring
- Broadcast model artifacts
- Throughput and latency tradeoffs

**Hands-on exercises**

- Score a large batch dataset with a Spark ML model.
- Use a pandas UDF to score records with a Python model.
- Write predictions to a Delta table.
- Add prediction quality checks.

**Real-world Data Engineering examples**

- Nightly churn score generation for all active customers.
- Scoring payment events for fraud signals in near real time.

**Recommended project after Module 13**

Build an ML feature and inference pipeline that creates customer features, trains a baseline model, runs batch inference, and writes predictions to a governed Delta table.

---

## Module 14: End-to-End Data Engineering Projects

**Goal:** Combine the full PySpark skill set into portfolio-grade systems.

### Chapter 14.1: Project Design and Delivery

#### Lesson 14.1.1: Requirements to Production Pipeline

**Learning objectives**

- Translate business requirements into data pipeline design.
- Select storage formats, partitioning, and processing patterns.
- Document assumptions, SLAs, and validation rules.

**Key concepts**

- Source analysis
- Data contracts
- Architecture diagrams
- Operational metrics
- Runbooks
- Cost and performance tradeoffs

**Hands-on exercises**

- Write a source-to-target mapping document.
- Design bronze, silver, and gold tables.
- Add observability metrics.
- Write a runbook for failed jobs.

**Real-world Data Engineering examples**

- Building a complete lakehouse pipeline for finance, retail, healthcare, or logistics data.
- Supporting BI dashboards and ML features from the same curated data platform.

**Recommended project after Module 14**

Design and implement a complete PySpark data platform for one business domain, including ingestion, validation, transformation, optimization, orchestration assumptions, tests, and production documentation.

---

## Final Capstone Projects

Choose one or more capstones depending on your target role.

### Capstone 1: E-Commerce Lakehouse Platform

Build a medallion architecture using orders, customers, payments, products, inventory, and clickstream data.

**Expected deliverables**

- Bronze ingestion for CSV, JSON, and Kafka events
- Silver cleaned entities with schema validation and deduplication
- Gold revenue, product, customer, and funnel metrics
- Delta Lake incremental loads and merge logic
- Tests for core transformations
- Spark UI optimization notes
- Deployment and runbook documentation

### Capstone 2: Real-Time Fraud Signal Pipeline

Build a streaming pipeline that ingests transaction events from Kafka and creates near-real-time fraud features.

**Expected deliverables**

- Kafka ingestion with Structured Streaming
- JSON parsing and validation
- Watermarked aggregations
- Quarantine handling for invalid events
- Delta output for clean transactions and features
- Batch inference or streaming scoring path
- Monitoring metrics and restart strategy

### Capstone 3: ML Feature Store Foundation

Build reusable feature tables for AI/ML teams.

**Expected deliverables**

- Customer, product, transaction, and behavioral feature tables
- Point-in-time feature generation
- Training dataset creation
- Batch inference dataset creation
- Data quality checks for freshness and completeness
- Documentation for feature definitions

### Capstone 4: Production Batch Pipeline Framework

Create a reusable PySpark project template for production jobs.

**Expected deliverables**

- Standard project structure
- SparkSession factory
- Config management
- Logging
- Data quality module
- Test fixtures
- CI/CD pipeline outline
- Example batch job with partition overwrite and backfill

---

## Interview Preparation

### Topics to Master

- Spark architecture: driver, executors, tasks, stages, and cluster managers
- Lazy evaluation and DAG execution
- DataFrames vs RDDs
- Transformations vs actions
- Narrow vs wide transformations
- Shuffle behavior
- Join strategies and broadcast joins
- Partitioning, bucketing, and file sizing
- Caching and persistence tradeoffs
- Spark SQL and Catalyst optimizer
- Schema enforcement and schema evolution
- Delta Lake transactions, merge, and time travel
- Structured Streaming, checkpoints, watermarks, and output modes
- Kafka offsets and streaming failure recovery
- Data quality validation
- Testing PySpark transformations
- Production deployment and CI/CD
- Feature engineering and model inference with Spark

### Interview Practice Plan

1. Explain every project in this roadmap using architecture, data flow, failure handling, and optimization decisions.
2. Practice reading Spark physical plans.
3. Debug sample scenarios involving skew, small files, driver OOM, executor OOM, and slow joins.
4. Write PySpark code from memory for joins, windows, aggregations, schema definitions, and Delta merge.
5. Prepare concise examples of production issues you can solve with Spark UI and logs.

---

## Common PySpark Interview Questions

1. What is PySpark, and how does it relate to Apache Spark?
2. What are the roles of driver and executors?
3. What is the difference between a job, stage, and task?
4. What is lazy evaluation in Spark?
5. What is the difference between transformations and actions?
6. What are narrow and wide transformations?
7. Why does `groupBy` usually cause a shuffle?
8. What is a SparkSession?
9. How do you define an explicit schema in PySpark?
10. Why is explicit schema usually better than schema inference in production?
11. When would you use CSV, JSON, Parquet, or Avro?
12. Why is Parquet commonly used in data lakes?
13. What are common write modes in Spark?
14. What is partition pruning?
15. What is bucketing, and when is it useful?
16. How do you handle small files in Spark?
17. What is a broadcast join?
18. How do you handle skewed joins?
19. What is the difference between left anti join and left semi join?
20. How do window functions work in Spark?
21. How would you deduplicate records using PySpark?
22. Why can Python UDFs be slow?
23. When would you use a pandas UDF?
24. How do you optimize a slow Spark job?
25. What information do you check in Spark UI?
26. What causes executor out-of-memory errors?
27. What causes driver out-of-memory errors?
28. How do caching and persistence work?
29. When should you avoid caching?
30. What is Delta Lake?
31. How does Delta Lake support ACID transactions?
32. What is Delta merge used for?
33. What is time travel in Delta Lake?
34. How do you design an incremental pipeline?
35. What is a high-water mark?
36. How do you handle late-arriving data?
37. What is Structured Streaming?
38. What are output modes in Structured Streaming?
39. What is checkpointing in streaming?
40. What is watermarking?
41. How does Spark integrate with Kafka?
42. How do you parse JSON messages from Kafka in Spark?
43. How do you test PySpark transformations?
44. What should a production PySpark project structure include?
45. How do you implement CI/CD for PySpark jobs?
46. How do you validate data quality in Spark?
47. How can PySpark support feature engineering?
48. What is a Spark ML pipeline?
49. How would you run model inference at scale with Spark?
50. How do you explain a PySpark project in a Data Engineering interview?

---

## Recommended Learning Resources

### Official Documentation

- Apache Spark Documentation: https://spark.apache.org/docs/latest/
- PySpark API Reference: https://spark.apache.org/docs/latest/api/python/
- Spark SQL Guide: https://spark.apache.org/docs/latest/sql-programming-guide.html
- Structured Streaming Guide: https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html
- Delta Lake Documentation: https://docs.delta.io/
- Apache Kafka Documentation: https://kafka.apache.org/documentation/

### Books

- *Learning Spark, 2nd Edition* by Jules S. Damji, Brooke Wenig, Tathagata Das, and Denny Lee
- *Spark: The Definitive Guide* by Bill Chambers and Matei Zaharia
- *Designing Data-Intensive Applications* by Martin Kleppmann
- *Fundamentals of Data Engineering* by Joe Reis and Matt Housley

### Practice Datasets

- NYC Taxi Trips
- Online Retail Dataset
- MovieLens
- Instacart Market Basket Analysis
- GitHub Archive
- Public cloud marketplace datasets

### Tools to Practice With

- PySpark local mode
- JupyterLab
- Docker Compose for Kafka
- Delta Lake
- pytest
- GitHub Actions
- Databricks Community Edition
- AWS EMR, GCP Dataproc, or Azure Synapse if cloud access is available

---

## Suggested Portfolio Sequence

1. Local PySpark file conversion pipeline
2. DataFrame and Spark SQL analytics mart
3. Customer 360 pipeline with joins and windows
4. Optimized Spark job with Spark UI analysis
5. Medallion data lake with Delta Lake
6. Incremental batch pipeline with Delta merge
7. Structured Streaming pipeline with Kafka
8. Tested production PySpark project template
9. ML feature engineering and batch inference pipeline
10. Full capstone lakehouse platform

