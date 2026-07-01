# Module 14: End-to-End Data Engineering Projects

**Goal:** Combine the full PySpark skill set into portfolio-grade systems.

## Learning Outcomes

By the end of this module, you will be able to:
- Translate business requirements into data pipeline design
- Select storage formats, partitioning, and processing patterns
- Document assumptions, SLAs, and validation rules
- Design and implement complete PySpark data platforms

---

## Chapter 14.1: Project Design and Delivery

### Lesson 14.1.1: Requirements to Production Pipeline

#### Source-to-Target Mapping

```python
# Document data flow
source_to_target = {
    "raw_orders": {
        "source": "s3://data-lake/raw/orders/",
        "format": "CSV",
        "target": "bronze/orders",
        "partition_by": "order_date",
        "schema_evolution": "add_columns"
    },
    "clean_orders": {
        "source": "bronze/orders",
        "format": "Parquet",
        "target": "silver/orders",
        "key": "order_id",
        "dedup_strategy": "latest_by_timestamp"
    },
    "daily_revenue": {
        "source": "silver/orders",
        "format": "Delta",
        "target": "gold/daily_revenue",
        "partition_by": "date",
        "refresh": "daily"
    }
}
```

#### Architecture Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Platform Architecture                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   Sources   │    │   Bronze    │    │   Silver    │    │
│  │             │    │             │    │             │    │
│  │ - CRM       │───►│ - Raw       │───►│ - Cleaned   │    │
│  │ - ERP       │    │ - Immutable │    │ - Validated │    │
│  │ - Events    │    │ - Partitioned│   │ - Deduped   │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                 │          │
│                                                 ▼          │
│                                        ┌─────────────┐    │
│                                        │    Gold     │    │
│                                        │             │    │
│                                        │ - Aggregated│    │
│                                        │ - Business  │    │
│                                        │ - Metrics   │    │
│                                        └─────────────┘    │
│                                                 │          │
│                                                 ▼          │
│                                        ┌─────────────┐    │
│                                        │  Consumers  │    │
│                                        │             │    │
│                                        │ - BI Tools  │    │
│                                        │ - ML Models │    │
│                                        │ - APIs      │    │
│                                        └─────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Operational Metrics

```python
# Track pipeline metrics
metrics = {
    "pipeline_name": "daily_orders",
    "execution_date": "2024-01-01",
    "start_time": start_time,
    "end_time": end_time,
    "duration_seconds": duration,
    "records_processed": record_count,
    "records_valid": valid_count,
    "records_invalid": invalid_count,
    "status": "success"
}

# Log to monitoring system
log_metrics(metrics)
```

#### Runbook

```markdown
# Pipeline Runbook: daily_orders

## Schedule
- Runs daily at 06:00 UTC
- SLA: Must complete by 08:00 UTC

## Inputs
- Source: s3://data-lake/raw/orders/
- Format: CSV
- Expected files: 1 per day

## Outputs
- Target: s3://data-lake/gold/daily_revenue/
- Format: Delta
- Partitioned by: date

## Dependencies
- upstream: raw_orders_ingestion
- downstream: revenue_dashboard, ml_training

## Failure Handling
1. Check source file existence
2. Verify record counts
3. Check for data quality issues
4. Restart from last successful partition

## Contacts
- Data Engineering: data-eng@company.com
- On-call: +1-555-0123
```

---

## Hands-On Exercises

### Exercise 1: Project Design

Design a complete PySpark data platform for a business domain:
1. Write a source-to-target mapping document
2. Design bronze, silver, and gold tables
3. Add observability metrics
4. Write a runbook for failed jobs

### Exercise 2: Implementation

Implement the designed platform:
1. Build all pipeline layers
2. Add data quality checks
3. Include tests
4. Document deployment process

---

## Recommended Project After Module 14

**Complete Data Platform**

Design and implement a complete PySpark data platform for one business domain, including:
- Ingestion
- Validation
- Transformation
- Optimization
- Orchestration assumptions
- Tests
- Production documentation

See `projects/project14_end_to_end/` for the complete implementation.
