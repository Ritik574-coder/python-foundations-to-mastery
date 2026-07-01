# Capstone 1: E-Commerce Lakehouse Platform

Build a medallion architecture using orders, customers, payments, products, inventory, and clickstream data.

## Project Overview

This capstone project combines all skills learned throughout the course to build a complete data lakehouse platform for an e-commerce business.

### Learning Objectives

- Design and implement a complete medallion architecture
- Handle multiple data sources and formats
- Implement incremental processing with Delta Lake
- Build production-quality data pipelines
- Create analytics-ready datasets

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    E-Commerce Lakehouse Platform                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Data Sources                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│
│  │  Orders  │  │ Customers│  │ Products │  │ Payments │  │Clickstrm ││
│  │  (CSV)   │  │  (CSV)   │  │  (CSV)   │  │  (CSV)   │  │  (JSON)  ││
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘│
│       │              │              │              │              │      │
│       ▼              ▼              ▼              ▼              ▼      │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                         Bronze Layer                                ││
│  │  Raw data ingestion with metadata                                   ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                  │                                      │
│                                  ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                         Silver Layer                                ││
│  │  Cleaned, validated, deduplicated entities                         ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                  │                                      │
│                                  ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                          Gold Layer                                 ││
│  │  Business aggregates and metrics                                    ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                  │                                      │
│                                  ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                       Consumers                                     ││
│  │  BI Dashboards │ ML Features │ APIs │ Reports                       ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Deliverables

### 1. Bronze Layer

- [ ] Raw ingestion for CSV, JSON, and Kafka events
- [ ] Schema validation at ingestion
- [ ] Metadata tracking (ingestion time, source file)
- [ ] Partitioning by ingestion date

### 2. Silver Layer

- [ ] Cleaned entities with schema validation
- [ ] Deduplication logic
- [ ] Data type standardization
- [ ] Foreign key validation

### 3. Gold Layer

- [ ] Revenue metrics (daily, weekly, monthly)
- [ ] Product analytics (top products, categories)
- [ ] Customer analytics (LTV, segments)
- [ ] Funnel metrics (conversion rates)

### 4. Delta Lake

- [ ] Incremental loads
- [ ] Merge/upsert logic
- [ ] Time travel support
- [ ] Schema evolution handling

### 5. Testing

- [ ] Unit tests for transformations
- [ ] Integration tests for pipelines
- [ ] Data quality checks

### 6. Documentation

- [ ] Architecture diagram
- [ ] Data dictionary
- [ ] Pipeline runbook
- [ ] Deployment guide

---

## Implementation

### Project Structure

```
capstone_ecommerce/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── io/
│   │   ├── __init__.py
│   │   ├── readers.py
│   │   └── writers.py
│   ├── bronze/
│   │   ├── __init__.py
│   │   └── ingest.py
│   ├── silver/
│   │   ├── __init__.py
│   │   ├── orders.py
│   │   ├── customers.py
│   │   └── products.py
│   ├── gold/
│   │   ├── __init__.py
│   │   ├── revenue.py
│   │   └── customer_analytics.py
│   └── utils/
│       ├── __init__.py
│       ├── spark.py
│       ├── validation.py
│       └── logging.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   └── integration/
├── configs/
│   ├── dev.yaml
│   └── prod.yaml
├── jobs/
│   ├── __init__.py
│   └── daily_pipeline.py
├── data/
│   └── sample_datasets/
├── requirements.txt
└── README.md
```

### Key Files

1. **config/settings.py** - Environment configurations
2. **io/readers.py** - Data source readers
3. **io/writers.py** - Data sink writers
4. **bronze/ingest.py** - Bronze layer ingestion
5. **silver/*.py** - Silver layer transformations
6. **gold/*.py** - Gold layer aggregations
7. **utils/validation.py** - Data quality checks
8. **jobs/daily_pipeline.py** - Main pipeline entry point

---

## Getting Started

1. Set up the development environment
2. Load sample data
3. Implement bronze layer
4. Implement silver layer
5. Implement gold layer
6. Add tests
7. Document the pipeline

---

## Evaluation Criteria

| Criteria | Weight | Description |
|----------|--------|-------------|
| Architecture | 25% | Clean separation of concerns, proper layering |
| Code Quality | 25% | Readable, maintainable, well-structured |
| Data Quality | 20% | Validation, error handling, idempotency |
| Testing | 15% | Unit and integration tests |
| Documentation | 15% | Clear docs, runbook, data dictionary |
