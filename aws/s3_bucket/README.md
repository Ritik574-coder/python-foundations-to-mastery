# AWS S3 Cheat Sheet — CLI, Python (boto3) & Data Lake Ops

## 1. Setup & Authentication

### Install tools
```bash
# AWS CLI
pip install awscli --break-system-packages
# or: sudo apt install awscli

# Python SDK
pip install boto3 --break-system-packages
```

### Configure credentials
```bash
aws configure
# Prompts for:
# AWS Access Key ID
# AWS Secret Access Key
# Default region (e.g. ap-south-1)
# Default output format (json)
```

This writes to `~/.aws/credentials` and `~/.aws/config`. You can also use named profiles:
```bash
aws configure --profile datalake
aws s3 ls --profile datalake
```

Environment variable alternative (useful in scripts/containers):
```bash
export AWS_ACCESS_KEY_ID=xxxx
export AWS_SECRET_ACCESS_KEY=xxxx
export AWS_DEFAULT_REGION=ap-south-1
```

boto3 automatically picks up credentials from `~/.aws/credentials`, env vars, or an IAM role if running on EC2/ECS/Lambda — no need to hardcode keys in code.

---

## 2. AWS CLI — Core S3 Commands

### Bucket operations
```bash
aws s3 ls                                  # list all buckets
aws s3 mb s3://my-bucket-name              # make bucket
aws s3 rb s3://my-bucket-name              # remove empty bucket
aws s3 rb s3://my-bucket-name --force      # remove bucket + all contents
```

### Listing objects
```bash
aws s3 ls s3://my-bucket-name              # list top-level
aws s3 ls s3://my-bucket-name/ --recursive # list all objects
aws s3 ls s3://my-bucket-name/ --recursive --human-readable --summarize
```

### Upload / Download
```bash
aws s3 cp file.csv s3://my-bucket-name/raw/file.csv
aws s3 cp s3://my-bucket-name/raw/file.csv ./file.csv
aws s3 cp ./local_folder s3://my-bucket-name/prefix/ --recursive
aws s3 cp s3://my-bucket-name/prefix/ ./local_folder --recursive
```

### Sync (only changed/new files — ideal for data lake ingestion)
```bash
aws s3 sync ./local_folder s3://my-bucket-name/prefix/
aws s3 sync s3://my-bucket-name/prefix/ ./local_folder
aws s3 sync ./local_folder s3://my-bucket-name/prefix/ --delete   # mirror, deletes extras
```

### Move / Remove
```bash
aws s3 mv file.csv s3://my-bucket-name/archive/file.csv
aws s3 rm s3://my-bucket-name/prefix/file.csv
aws s3 rm s3://my-bucket-name/prefix/ --recursive   # delete a "folder"
```

### Presigned URL (temporary shareable link)
```bash
aws s3 presign s3://my-bucket-name/file.csv --expires-in 3600
```

### Storage class & encryption on upload
```bash
aws s3 cp file.csv s3://my-bucket-name/ --storage-class STANDARD_IA
aws s3 cp file.csv s3://my-bucket-name/ --sse AES256
```

### Bucket policy / versioning / lifecycle (via `s3api`)
```bash
aws s3api put-bucket-versioning --bucket my-bucket-name \
  --versioning-configuration Status=Enabled

aws s3api get-bucket-versioning --bucket my-bucket-name

aws s3api put-bucket-lifecycle-configuration --bucket my-bucket-name \
  --lifecycle-configuration file://lifecycle.json

aws s3api put-bucket-policy --bucket my-bucket-name \
  --policy file://policy.json
```

---

## 3. Python (boto3) — S3 Client Basics

### Setup
```python
import boto3

# Uses default profile / env vars / IAM role automatically
s3 = boto3.client("s3")

# Or with an explicit profile
session = boto3.Session(profile_name="datalake")
s3 = session.client("s3")

# Or explicit keys (avoid hardcoding in real projects — use env vars/secrets manager)
s3 = boto3.client(
    "s3",
    aws_access_key_id="xxxx",
    aws_secret_access_key="xxxx",
    region_name="ap-south-1",
)
```

### List buckets and objects
```python
buckets = s3.list_buckets()
for b in buckets["Buckets"]:
    print(b["Name"])

response = s3.list_objects_v2(Bucket="my-bucket-name", Prefix="raw/")
for obj in response.get("Contents", []):
    print(obj["Key"], obj["Size"], obj["LastModified"])
```

### Paginate through large buckets (important for real data lakes)
```python
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket="my-bucket-name", Prefix="raw/"):
    for obj in page.get("Contents", []):
        print(obj["Key"])
```

### Upload files
```python
s3.upload_file("local_file.csv", "my-bucket-name", "raw/local_file.csv")

# With extra args (storage class, encryption, content type)
s3.upload_file(
    "local_file.csv", "my-bucket-name", "raw/local_file.csv",
    ExtraArgs={"StorageClass": "STANDARD_IA", "ServerSideEncryption": "AES256"}
)
```

### Download files
```python
s3.download_file("my-bucket-name", "raw/local_file.csv", "local_file.csv")
```

### Upload/download in-memory (no local disk write — great for pipelines)
```python
import io
import pandas as pd

# Upload a DataFrame directly as CSV
buffer = io.StringIO()
df.to_csv(buffer, index=False)
s3.put_object(Bucket="my-bucket-name", Key="processed/data.csv", Body=buffer.getvalue())

# Read a CSV straight into a DataFrame
obj = s3.get_object(Bucket="my-bucket-name", Key="raw/data.csv")
df = pd.read_csv(io.BytesIO(obj["Body"].read()))
```

### Delete objects
```python
s3.delete_object(Bucket="my-bucket-name", Key="raw/old_file.csv")

# Batch delete (up to 1000 keys per call)
s3.delete_objects(
    Bucket="my-bucket-name",
    Delete={"Objects": [{"Key": "a.csv"}, {"Key": "b.csv"}]}
)
```

### Copy / move between prefixes or buckets
```python
s3.copy_object(
    Bucket="my-bucket-name",
    CopySource={"Bucket": "my-bucket-name", "Key": "raw/file.csv"},
    Key="archive/file.csv",
)
# "Move" = copy + delete original
s3.delete_object(Bucket="my-bucket-name", Key="raw/file.csv")
```

### Generate a presigned URL
```python
url = s3.generate_presigned_url(
    "get_object",
    Params={"Bucket": "my-bucket-name", "Key": "raw/file.csv"},
    ExpiresIn=3600,
)
```

### Check if an object exists (common pattern)
```python
from botocore.exceptions import ClientError

def object_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise
```

---

## 4. Higher-Level Resource API (more Pythonic)

```python
s3_resource = boto3.resource("s3")
bucket = s3_resource.Bucket("my-bucket-name")

for obj in bucket.objects.filter(Prefix="raw/"):
    print(obj.key, obj.size)

bucket.upload_file("local.csv", "raw/local.csv")
bucket.download_file("raw/local.csv", "local.csv")
```

---

## 5. Data Lake / Lakehouse Patterns on S3

### Typical folder (prefix) layout
```
s3://my-datalake/
  raw/           <- landing zone, untouched source data
  bronze/        <- validated/deduped raw data
  silver/        <- cleaned, typed, joined
  gold/          <- aggregated, business-ready
```

### Partitioning (critical for query performance with Athena/Spark/Glue)
```
s3://my-datalake/silver/sales/year=2026/month=07/day=06/part-0001.parquet
```
```python
key = f"silver/sales/year={year}/month={month:02d}/day={day:02d}/part-0001.parquet"
s3.put_object(Bucket="my-datalake", Key=key, Body=parquet_bytes)
```

### Writing Parquet directly to S3 (preferred format for lakehouses)
```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import io

table = pa.Table.from_pandas(df)
buf = io.BytesIO()
pq.write_table(table, buf)
s3.put_object(Bucket="my-datalake", Key="silver/sales/part-0001.parquet", Body=buf.getvalue())
```

### Reading Parquet back
```python
obj = s3.get_object(Bucket="my-datalake", Key="silver/sales/part-0001.parquet")
df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
```

### Using `s3fs` for pandas-native S3 paths (very common in lakehouse code)
```bash
pip install s3fs --break-system-packages
```
```python
df.to_parquet("s3://my-datalake/silver/sales/part-0001.parquet")
df = pd.read_parquet("s3://my-datalake/silver/sales/part-0001.parquet")
```

### Multipart upload for very large files (automatic via `upload_file`, manual control below)
```python
from boto3.s3.transfer import TransferConfig

config = TransferConfig(multipart_threshold=1024 * 25, max_concurrency=10)
s3.upload_file("big_file.parquet", "my-datalake", "raw/big_file.parquet", Config=config)
```

### Event-driven ingestion (trigger a Lambda when new data lands)
```bash
aws s3api put-bucket-notification-configuration --bucket my-datalake \
  --notification-configuration file://notification.json
```

---

## 6. Handy One-Liners

```bash
# Bucket size + object count
aws s3 ls s3://my-bucket-name --recursive --summarize | tail -2

# Find files larger than X (requires jq)
aws s3api list-objects-v2 --bucket my-bucket-name --query "Contents[?Size>\`104857600\`]"

# Empty a bucket fast (versioned buckets need extra steps)
aws s3 rm s3://my-bucket-name --recursive
```

---

## 7. Quick Reference Table

| Task                     | CLI                                  | Python (boto3)                          |
|--------------------------|---------------------------------------|-------------------------------------------|
| List buckets             | `aws s3 ls`                          | `s3.list_buckets()`                       |
| List objects             | `aws s3 ls s3://b/ --recursive`      | `s3.list_objects_v2(Bucket=b)`             |
| Upload file               | `aws s3 cp f s3://b/f`               | `s3.upload_file(f, b, key)`                |
| Download file             | `aws s3 cp s3://b/f f`               | `s3.download_file(b, key, f)`              |
| Sync folder                | `aws s3 sync ./d s3://b/p/`         | (no direct equiv — loop + `upload_file`)   |
| Delete object              | `aws s3 rm s3://b/f`                 | `s3.delete_object(Bucket=b, Key=key)`      |
| Copy/move                  | `aws s3 mv/cp`                       | `s3.copy_object(...)`                      |
| Presigned URL               | `aws s3 presign s3://b/f`           | `s3.generate_presigned_url(...)`           |
| Create bucket               | `aws s3 mb s3://b`                   | `s3.create_bucket(Bucket=b)`               |

---

**Notes / gotchas:**
- S3 has no real "folders" — prefixes just look like folders in keys (e.g. `raw/file.csv`).
- `sync` is your best friend for incremental data lake ingestion; it skips unchanged files.
- Always use Parquet (not CSV) for anything you'll query at scale (Athena, Spark, DuckDB) — smaller, columnar, faster.
- Use IAM roles instead of access keys whenever running on AWS infrastructure (EC2, Lambda, Glue) — no keys to leak.
- For huge datasets, prefer `s3fs` + pandas or PyArrow's native S3 filesystem over manual `get_object`/`put_object` loops.