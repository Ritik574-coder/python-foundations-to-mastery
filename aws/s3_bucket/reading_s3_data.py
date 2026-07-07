# Import Required module 
import os 
import pandas as pd
import logging 

# Logger initializeation
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


#AWS authentication configuration
try: 
    os.environ["AWS_ACCESS_KEY"] = os.getenv("AWS_ACCESS_KEY")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
    os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_REGION")

    logger.info("AWS authentication configuration loaded successfully.")

except Exception as error:
    logger.exception(
        "Error occurred while loading AWS authentication configuration."
    )
    raise


# fetched  customer data from s3 bucket 
try:
    dfc = pd.read_csv("s3://ritsky-bucket-2025/retail_raw_dataset/raw_customers.csv")
    logger.info("Data fetched successfully from the S3 bucket.")

except Exception as error:
    logger.exception("Failed to fetch data from the S3 bucket.")


# count the total row and column of source dataset 
logger.info(
    f"Source dataset contains {dfc.shape[0]} records and {dfc.shape[1]} columns."
)


# breaking dataset using domain logic
customer_identity = [
    'customer_id', 'title',
    'first_name' , 'last_name', 
    'gender'     , 'is_active'
]
customer_address = [
    'customer_id'  , 'address', 
    'city', 'state', 'state_abbr', 
    'zip_code'     , 'country', 
    'region'
]
customer_content = [
    'customer_id', 'email', 
    'phone'
]
customer_business_info = [
    'customer_id'       , 'customer_segment', 
    'loyalty_points'    , 'preferred_channel', 
    'annual_income_usd' , 'company' 
]
customer_dates = [
    'customer_id', 'date_of_birth', 
    'account_created_date'
]

# Creating DataFrame for eatch domain 
try :
    customer_identity_df = dfc[customer_identity]
    customer_address_df = dfc[customer_address]
    customer_content_df = dfc[customer_content]
    customer_business_info_df = dfc[customer_business_info]
    customer_dates_df = dfc[customer_dates]

    logger.info("Customer domain DataFrames created successfully.")

    logger.info(f"customer_identity_df domain contains {customer_identity_df.shape[0]} record and {customer_identity_df.shape[1]} columns.")
    logger.info(f"customer_address_df domain contains {customer_address_df.shape[0]} record and {customer_address_df.shape[1]} columns.")
    logger.info(f"customer_content_df domain contains {customer_content_df.shape[0]} record and {customer_content_df.shape[1]} columns.")
    logger.info(f"customer_business_info_df domain contains {customer_business_info_df.shape[0]} record and {customer_business_info_df.shape[1]} columns.")
    logger.info(f"customer_dates_df domain contains {customer_dates_df.shape[0]} record and {customer_dates_df.shape[1]} columns.")

except Exception as error : 
    logger.exception("Failed to create customer domain DataFrames.")
    raise


required_columns = (
    customer_identity
    + customer_address
    + customer_content
    + customer_business_info
    + customer_dates
)

missing_columns = set(required_columns) - set(dfc.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )
else : 
    logger.info("All DataFrame column match to required_columns.")