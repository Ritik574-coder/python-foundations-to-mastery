# Module 12: Production Deployment and CI/CD

**Goal:** Deploy PySpark jobs reliably and automate quality checks before release.

## Learning Outcomes

By the end of this module, you will be able to:
- Compare deployment targets for Spark
- Submit jobs with dependencies and configuration
- Build CI checks for PySpark repositories
- Promote jobs between environments

---

## Chapter 12.1: Production Deployment

### Lesson 12.1.1: Running Spark Jobs in Real Environments

#### Deployment Targets

| Platform | Description | Best For |
|----------|-------------|----------|
| **spark-submit** | Local or cluster | Simple deployments |
| **Databricks** | Managed Spark platform | Production, collaboration |
| **EMR** | AWS managed Hadoop | AWS ecosystem |
| **Dataproc** | GCP managed Spark | GCP ecosystem |
| **Kubernetes** | Container orchestration | Cloud-native |

#### spark-submit

```bash
# Basic spark-submit
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --driver-memory 4g \
    --executor-memory 8g \
    --executor-cores 4 \
    --num-executors 10 \
    --jars dependencies.jar \
    jobs/daily_etl.py \
    --execution-date 2024-01-01
```

#### Databricks Jobs

```python
# Databricks job configuration
job_config = {
    "run_name": "Daily ETL",
    "new_cluster": {
        "spark_version": "12.2.x-scala2.12",
        "node_type_id": "i3.xlarge",
        "num_workers": 4
    },
    "notebook_task": {
        "notebook_path": "/Repos/project/jobs/daily_etl"
    },
    "schedule": {
        "quartz_cron_expression": "0 0 6 * * ?",
        "timezone_id": "America/New_York"
    }
}
```

#### Secrets Management

```python
# Databricks
password = dbutils.secrets.get(scope="my-scope", key="db-password")

# Environment variables
import os
password = os.environ.get("DB_PASSWORD")

# Spark config
spark.conf.get("spark.secret.password")
```

---

## Chapter 12.2: CI/CD for PySpark

### Lesson 12.2.1: Automated Checks and Releases

#### CI Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pyspark
      
      - name: Run linting
        run: |
          black --check src/
          flake8 src/
      
      - name: Run tests
        run: pytest tests/ -v
```

#### CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to Databricks
        uses: databricks/databricks-cli-action@v1
        with:
          command: workspace import_dir src/ /Repos/project/src
```

#### Testing Strategy

| Test Type | When | What to Test |
|-----------|------|--------------|
| **Unit** | Every commit | Individual functions |
| **Integration** | Every PR | Full pipeline |
| **Performance** | Before release | Latency, throughput |
| **Smoke** | After deploy | Critical paths |

---

## Hands-On Exercises

### Exercise 1: Deployable Job

Create a deployable job that:
1. Accepts runtime parameters
2. Packages dependencies
3. Simulates development and production configs

### Exercise 2: CI/CD Pipeline

Build a CI/CD pipeline that:
1. Runs pytest test execution
2. Adds code formatting and linting checks
3. Builds a deployable artifact
4. Creates a deployment checklist

---

## Recommended Project After Module 12

**CI/CD-Ready Repository**

Build a CI/CD-ready PySpark repository with:
- Tests
- Linting
- Environment configs
- Packaged jobs
- Deployment guide for Databricks, EMR, or Kubernetes

See `projects/project12_production_deployment/` for the complete implementation.
