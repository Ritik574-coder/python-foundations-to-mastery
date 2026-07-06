# AWS RDS Cheat Sheet — CLI, Python (boto3) & DB Connections

## 1. Setup & Authentication

Same AWS setup as S3 — RDS *management* (create/stop/snapshot instances) uses the AWS CLI/boto3 with your IAM credentials. *Connecting to the database itself* (running queries) is separate and uses a DB driver (psycopg2, pymysql, etc.) with DB credentials, not IAM keys.

```bash
pip install awscli boto3 --break-system-packages

# DB drivers (pick based on engine)
pip install psycopg2-binary --break-system-packages   # PostgreSQL
pip install pymysql --break-system-packages           # MySQL/MariaDB
pip install sqlalchemy --break-system-packages         # ORM / engine-agnostic layer
```

```bash
aws configure --profile rds-admin
```

---

## 2. AWS CLI — Core RDS Commands

### List & describe instances
```bash
aws rds describe-db-instances
aws rds describe-db-instances --db-instance-identifier mydb --output table
aws rds describe-db-instances --query "DBInstances[*].{ID:DBInstanceIdentifier,Status:DBInstanceStatus,Endpoint:Endpoint.Address}"
```

### Create an instance
```bash
aws rds create-db-instance \
  --db-instance-identifier mydb \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username admin \
  --master-user-password 'SuperSecret123!' \
  --allocated-storage 20 \
  --vpc-security-group-ids sg-0123456789abcdef0 \
  --publicly-accessible
```

### Start / Stop / Reboot / Delete
```bash
aws rds stop-db-instance --db-instance-identifier mydb
aws rds start-db-instance --db-instance-identifier mydb
aws rds reboot-db-instance --db-instance-identifier mydb

aws rds delete-db-instance --db-instance-identifier mydb \
  --skip-final-snapshot   # or use --final-db-snapshot-identifier mydb-final
```

### Modify an instance (resize, change storage, etc.)
```bash
aws rds modify-db-instance \
  --db-instance-identifier mydb \
  --db-instance-class db.t3.small \
  --apply-immediately
```

### Snapshots (backups)
```bash
aws rds create-db-snapshot \
  --db-instance-identifier mydb \
  --db-snapshot-identifier mydb-snapshot-2026-07-06

aws rds describe-db-snapshots --db-instance-identifier mydb

aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier mydb-restored \
  --db-snapshot-identifier mydb-snapshot-2026-07-06

aws rds delete-db-snapshot --db-snapshot-identifier mydb-snapshot-2026-07-06
```

### Point-in-time restore
```bash
aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier mydb \
  --target-db-instance-identifier mydb-pitr \
  --restore-time 2026-07-01T04:00:00Z
```

### Security groups & networking
```bash
aws rds describe-db-instances --db-instance-identifier mydb \
  --query "DBInstances[0].VpcSecurityGroups"

aws ec2 authorize-security-group-ingress \
  --group-id sg-0123456789abcdef0 \
  --protocol tcp --port 5432 --cidr YOUR_IP/32
```

### Read replicas
```bash
aws rds create-db-instance-read-replica \
  --db-instance-identifier mydb-replica \
  --source-db-instance-identifier mydb
```

### Logs & events
```bash
aws rds describe-db-log-files --db-instance-identifier mydb
aws rds download-db-log-file-portion --db-instance-identifier mydb \
  --log-file-name error/postgresql.log --output text

aws rds describe-events --source-identifier mydb --source-type db-instance
```

### Parameter & option groups (engine config tuning)
```bash
aws rds describe-db-parameter-groups
aws rds create-db-parameter-group --db-parameter-group-name my-pg-params \
  --db-parameter-group-family postgres15 --description "custom params"
aws rds modify-db-parameter-group --db-parameter-group-name my-pg-params \
  --parameters "ParameterName=work_mem,ParameterValue=16384,ApplyMethod=immediate"
```

---

## 3. Python (boto3) — RDS Management API

boto3's RDS client wraps the same actions as the CLI — useful for automation scripts (e.g. spin up/down dev databases, scheduled snapshotting).

```python
import boto3

rds = boto3.client("rds", region_name="ap-south-1")
```

### List instances
```python
response = rds.describe_db_instances()
for db in response["DBInstances"]:
    print(db["DBInstanceIdentifier"], db["DBInstanceStatus"], db["Endpoint"]["Address"])
```

### Create an instance
```python
rds.create_db_instance(
    DBInstanceIdentifier="mydb",
    DBInstanceClass="db.t3.micro",
    Engine="postgres",
    MasterUsername="admin",
    MasterUserPassword="SuperSecret123!",
    AllocatedStorage=20,
    VpcSecurityGroupIds=["sg-0123456789abcdef0"],
    PubliclyAccessible=True,
)
```

### Start / stop / delete
```python
rds.stop_db_instance(DBInstanceIdentifier="mydb")
rds.start_db_instance(DBInstanceIdentifier="mydb")
rds.delete_db_instance(DBInstanceIdentifier="mydb", SkipFinalSnapshot=True)
```

### Wait until available (useful in scripts — avoids polling manually)
```python
waiter = rds.get_waiter("db_instance_available")
waiter.wait(DBInstanceIdentifier="mydb")
print("DB is ready")
```

### Snapshots
```python
rds.create_db_snapshot(
    DBInstanceIdentifier="mydb",
    DBSnapshotIdentifier="mydb-snapshot-2026-07-06",
)

snapshots = rds.describe_db_snapshots(DBInstanceIdentifier="mydb")
for s in snapshots["DBSnapshots"]:
    print(s["DBSnapshotIdentifier"], s["Status"])
```

### Get connection details programmatically (handy before connecting)
```python
db = rds.describe_db_instances(DBInstanceIdentifier="mydb")["DBInstances"][0]
host = db["Endpoint"]["Address"]
port = db["Endpoint"]["Port"]
engine = db["Engine"]
print(host, port, engine)
```

---

## 4. Connecting to RDS & Running Queries (this is NOT boto3 — use a DB driver)

### PostgreSQL — psycopg2
```python
import psycopg2

conn = psycopg2.connect(
    host="mydb.xxxxxxxxxx.ap-south-1.rds.amazonaws.com",
    port=5432,
    dbname="postgres",
    user="admin",
    password="SuperSecret123!",
)
cur = conn.cursor()
cur.execute("SELECT * FROM sales LIMIT 10;")
rows = cur.fetchall()
for row in rows:
    print(row)
cur.close()
conn.close()
```

### MySQL — pymysql
```python
import pymysql

conn = pymysql.connect(
    host="mydb.xxxxxxxxxx.ap-south-1.rds.amazonaws.com",
    port=3306,
    user="admin",
    password="SuperSecret123!",
    database="mydb",
)
with conn.cursor() as cur:
    cur.execute("SELECT * FROM sales LIMIT 10;")
    for row in cur.fetchall():
        print(row)
conn.close()
```

### SQLAlchemy (engine-agnostic — works with pandas too)
```python
from sqlalchemy import create_engine
import pandas as pd

# Postgres
engine = create_engine("postgresql+psycopg2://admin:SuperSecret123!@mydb.xxxx.rds.amazonaws.com:5432/postgres")

# MySQL
# engine = create_engine("mysql+pymysql://admin:SuperSecret123!@mydb.xxxx.rds.amazonaws.com:3306/mydb")

df = pd.read_sql("SELECT * FROM sales", engine)
df.to_sql("sales_copy", engine, if_exists="replace", index=False)
```

### IAM authentication (no password — token-based, more secure)
```python
import boto3

rds_client = boto3.client("rds", region_name="ap-south-1")
token = rds_client.generate_db_auth_token(
    DBHostname="mydb.xxxx.rds.amazonaws.com",
    Port=5432,
    DBUsername="iam_user",
)

conn = psycopg2.connect(
    host="mydb.xxxx.rds.amazonaws.com",
    port=5432,
    dbname="postgres",
    user="iam_user",
    password=token,
    sslmode="require",
)
```
*(Requires IAM DB authentication enabled on the instance and the DB user granted the `rds_iam` role/permission.)*

---

## 5. RDS + S3 (common lakehouse/ETL pattern)

### Export a query result to S3 as Parquet
```python
import pandas as pd
import boto3, io, pyarrow as pa, pyarrow.parquet as pq

df = pd.read_sql("SELECT * FROM sales", engine)

table = pa.Table.from_pandas(df)
buf = io.BytesIO()
pq.write_table(table, buf)

s3 = boto3.client("s3")
s3.put_object(Bucket="my-datalake", Key="silver/sales/export.parquet", Body=buf.getvalue())
```

### Native RDS → S3 export (no code, uses AWS-managed export)
```bash
aws rds start-export-task \
  --export-task-identifier sales-export \
  --source-arn arn:aws:rds:ap-south-1:123456789012:snapshot:mydb-snapshot-2026-07-06 \
  --s3-bucket-name my-datalake \
  --iam-role-arn arn:aws:iam::123456789012:role/rds-s3-export-role \
  --kms-key-id arn:aws:kms:ap-south-1:123456789012:key/xxxx
```

### Loading data from S3 into RDS (PostgreSQL `aws_s3` extension)
```sql
CREATE EXTENSION aws_s3 CASCADE;
SELECT aws_s3.table_import_from_s3(
  'sales',
  '',
  '(format csv, header true)',
  aws_commons.create_s3_uri('my-datalake', 'raw/sales.csv', 'ap-south-1')
);
```

---

## 6. Quick Reference Table

| Task                      | CLI                                          | Python (boto3 / driver)                     |
|---------------------------|-----------------------------------------------|-----------------------------------------------|
| List instances            | `aws rds describe-db-instances`              | `rds.describe_db_instances()`                 |
| Create instance            | `aws rds create-db-instance ...`             | `rds.create_db_instance(...)`                  |
| Start/stop instance         | `aws rds start/stop-db-instance`            | `rds.start_db_instance(...)`                   |
| Delete instance              | `aws rds delete-db-instance ...`           | `rds.delete_db_instance(...)`                  |
| Create snapshot               | `aws rds create-db-snapshot ...`          | `rds.create_db_snapshot(...)`                  |
| Restore snapshot                | `aws rds restore-db-instance-from-db-snapshot` | `rds.restore_db_instance_from_db_snapshot(...)` |
| Run a query                       | (use `psql`/`mysql` CLI client)          | `psycopg2`/`pymysql`/SQLAlchemy               |
| Export data to S3                    | `aws rds start-export-task ...`      | manual: query → pandas → `s3.put_object`      |

---

**Notes / gotchas:**
- Two different "worlds": **boto3 `rds` client** manages the *infrastructure* (create/stop/snapshot); a **DB driver** (psycopg2/pymysql) manages the *data* (queries, tables). Don't confuse them.
- Always restrict your security group ingress to your IP/VPC — don't leave `0.0.0.0/0` open on the DB port.
- Prefer IAM database authentication over static passwords where possible — no credentials to rotate or leak.
- `db.t3.micro`/`db.t4g.micro` are the free-tier-eligible instance classes for learning/testing.
- Stopping an RDS instance doesn't stop billing for storage, and AWS auto-restarts a stopped instance after ~7 days — good to know if you're experimenting to save cost.
- Use `psql -h <endpoint> -U admin -d postgres` or `mysql -h <endpoint> -u admin -p` from the terminal for quick manual checks without writing Python.