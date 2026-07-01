# PySpark Course Projects

This directory contains standalone project implementations for each module.

## Project List

| # | Project | Module | Description |
|---|---------|--------|-------------|
| 1 | Orders Pipeline | 1 | Local PySpark job reading orders, filtering, and writing Parquet |
| 2 | Diagnostic App | 2 | Small diagnostic app documenting jobs, stages, tasks |
| 3 | Raw to Curated | 3 | File conversion pipeline with explicit schemas |
| 4 | Analytics Mart | 4 | Revenue, products, and customer analytics using DataFrame API and SQL |
| 5 | Customer 360 | 5 | Multi-source joins with window functions |
| 6 | Optimization | 6 | Optimized Spark job with Spark UI analysis |
| 7 | Medallion Lake | 7 | Bronze, silver, gold data lake architecture |
| 8 | Delta Incremental | 8 | Incremental Delta Lake pipeline with merge |
| 9 | Batch Production | 9 | Production-style batch pipeline with backfill |
| 10 | Streaming Kafka | 10 | Structured Streaming with Kafka |
| 11 | Production Template | 11 | Reusable PySpark project template |
| 12 | CI/CD Ready | 12 | CI/CD-ready PySpark repository |
| 13 | ML Feature Store | 13 | ML feature engineering and inference pipeline |
| 14 | End-to-End | 14 | Complete data platform implementation |

## Capstone Projects

| # | Capstone | Description |
|---|----------|-------------|
| 1 | E-Commerce Lakehouse | Complete medallion architecture for e-commerce |
| 2 | Fraud Detection | Real-time fraud signal pipeline |
| 3 | Feature Store | ML feature store foundation |
| 4 | Batch Framework | Reusable batch pipeline framework |

## Getting Started

Each project directory contains:
- `README.md` - Project description and requirements
- `src/` - Source code
- `tests/` - Test files
- `configs/` - Configuration files

### Running a Project

```bash
# Navigate to project directory
cd projects/project01_orders_pipeline

# Install dependencies (if needed)
pip install -r requirements.txt

# Run the project
python src/main.py
```

### Project Structure

Each project follows a consistent structure:

```
projectXX_name/
├── src/
│   ├── __init__.py
│   ├── main.py
│   └── ...
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── configs/
│   └── config.yaml
├── requirements.txt
└── README.md
```

## Learning Path

Complete projects in order as you progress through the modules:

1. **Module 1-3**: Projects 1-3 (Foundations)
2. **Module 4-6**: Projects 4-6 (Core Data Engineering)
3. **Module 7-8**: Projects 7-8 (Lakehouse)
4. **Module 9-10**: Projects 9-10 (Production & Streaming)
5. **Module 11-12**: Projects 11-12 (Quality & Deployment)
6. **Module 13-14**: Projects 13-14 & Capstones (ML & End-to-End)

## Tips

1. Start with the basics and build up
2. Review the module README before starting the project
3. Run the project code and experiment with modifications
4. Check the solutions directory (if available) after attempting
5. Apply learnings to your own data and use cases
