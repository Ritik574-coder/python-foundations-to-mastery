# PySpark Complete Course: From Foundations to Production

> A comprehensive hands-on course for Data Engineers building production Spark pipelines and supporting AI/ML workloads.

## Course Structure

This course is organized into 14 modules, each building on the previous one:

| Module | Title | Focus |
|--------|-------|-------|
| 01 | PySpark Fundamentals | Getting started, first applications |
| 02 | Spark Architecture | Driver, executors, DAGs, SparkSession |
| 03 | DataFrames, Schemas & File Formats | Data representation and I/O |
| 04 | Transformations, Actions & SQL | Core data processing |
| 05 | Joins, Windows & Advanced Processing | Relational and time-series operations |
| 06 | Performance Optimization | Tuning and debugging |
| 07 | ETL Pipelines & Data Lake | Pipeline design and medallion architecture |
| 08 | Delta Lake & Incremental Processing | ACID transactions and CDC |
| 09 | Batch Processing in Production | Scheduled jobs and backfills |
| 10 | Structured Streaming & Kafka | Real-time data processing |
| 11 | Data Quality & Testing | Validation and testability |
| 12 | Production Deployment & CI/CD | Deployment and automation |
| 13 | Machine Learning Workloads | Feature engineering and inference |
| 14 | End-to-End Projects | Portfolio-grade systems |

## Directory Structure

```
PySpark/
├── course/                     # Main course content
│   ├── module01_pyspark_fundamentals/
│   ├── module02_spark_architecture/
│   ├── ...
│   └── module14_end_to_end_projects/
├── projects/                   # Standalone project implementations
│   ├── project01_orders_pipeline/
│   ├── ...
│   └── capstone_ecommerce/
├── data/                       # Sample datasets
│   └── sample_datasets/
└── tests/                      # Course tests
```

## Getting Started

### Prerequisites

- Python 3.10+
- Java 8 or 11 (compatible with your Spark version)
- PySpark 3.x or 4.x

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install pyspark pytest pandas

# For Delta Lake (Module 8+)
pip install delta-spark

# For Kafka integration (Module 10)
pip install pyspark[kafka]
```

### Running Examples

Each module contains:
- `README.md` - Theory and concepts
- `scripts/` - Hands-on Python exercises
- `data/` - Sample data files

```bash
# Run a module script
cd course/module01_pyspark_fundamentals
python scripts/01_what_pyspark_solves.py

# Run tests
cd PySpark
pytest tests/
```

## Learning Path

```
Foundations → Core Data Engineering → Optimization → Lakehouse → Streaming → Production → AI/ML
```

**Time Commitment:**
- Beginner: 8-10 hours/week for 8-10 weeks
- Intermediate: 10-15 hours/week for 10-12 weeks
- Advanced: 15+ weeks with production-style projects

## Sample Datasets

The `data/sample_datasets/` directory contains:
- `customers.csv` - Customer master data
- `orders.csv` - Transactional orders
- `products.csv` - Product catalog
- `payments.csv` - Payment transactions
- `clickstream.json` - Web click events
- `transactions.json` - Payment transactions (nested)

## Course Projects

After each module, complete the recommended project in the `projects/` directory. Final capstone projects combine all skills into portfolio-grade systems.

## Interview Preparation

Each module includes interview-relevant concepts. The final sections provide:
- 50 common PySpark interview questions
- Practice plan for technical interviews
- Topics to master for Data Engineering roles
