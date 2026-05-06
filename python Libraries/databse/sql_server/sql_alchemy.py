import os
import time
import logging

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

DB_SERVER = os.getenv("DB_SERVER")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# =========================================================
# VALIDATE ENV VARIABLES
# =========================================================

required_env_vars = {
    "DB_SERVER": DB_SERVER,
    "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD
}

missing_vars = [
    key for key, value in required_env_vars.items()
    if not value
]

if missing_vars:
    raise ValueError(
        f"Missing environment variables: {', '.join(missing_vars)}"
    )

# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# =========================================================
# CREATE SQLALCHEMY ENGINE
# =========================================================

def create_db_engine(database: str = "master") -> Engine:

    try:

        connection_url = (
            f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}"
            f"@{DB_SERVER}/{database}"
            "?driver=ODBC+Driver+18+for+SQL+Server"
            "&TrustServerCertificate=yes"
        )

        logger.info(f"Creating engine for database: {database}")

        engine = create_engine(
            connection_url,

            # Pool settings
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,

            # SQLAlchemy 2.0 style
            future=True
        )

        logger.info("SQLAlchemy engine created successfully")

        return engine

    except SQLAlchemyError:
        logger.exception("Failed to create SQLAlchemy engine")
        raise

# =========================================================
# CREATE DATABASE
# =========================================================

def create_database(database_name: str = "PythonDB") -> None:

    try:

        engine = create_db_engine("master")

        with engine.connect() as conn:

            # IMPORTANT
            conn = conn.execution_options(
                isolation_level="AUTOCOMMIT"
            )

            logger.info(
                f"Dropping database if exists: {database_name}"
            )

            conn.execute(text(f"""
                IF EXISTS (
                    SELECT name
                    FROM sys.databases
                    WHERE name = '{database_name}'
                )
                BEGIN

                    ALTER DATABASE [{database_name}]
                    SET SINGLE_USER
                    WITH ROLLBACK IMMEDIATE;

                    DROP DATABASE [{database_name}];

                END;
            """))

            logger.info(
                f"Creating database: {database_name}"
            )

            conn.execute(text(f"""
                CREATE DATABASE [{database_name}]
            """))

            logger.info(
                f"Database created successfully: {database_name}"
            )

    except SQLAlchemyError:
        logger.exception("Database creation failed")
        raise

# =========================================================
# CREATE SCHEMAS
# =========================================================

def create_schemas(database_name: str = "PythonDB") -> None:

    schemas = ["bronze", "silver", "gold"]

    try:

        engine = create_db_engine(database_name)

        with engine.begin() as conn:

            for schema in schemas:

                logger.info(
                    f"Creating schema: {schema}"
                )

                conn.execute(text(f"""
                    IF NOT EXISTS (
                        SELECT *
                        FROM sys.schemas
                        WHERE name = '{schema}'
                    )
                    BEGIN
                        EXEC(
                            'CREATE SCHEMA {schema} AUTHORIZATION dbo'
                        );
                    END;
                """))

            logger.info(
                "All schemas created successfully"
            )

    except SQLAlchemyError:
        logger.exception("Schema creation failed")
        raise

# =========================================================
# MAIN
# =========================================================

def main() -> None:

    DATABASE_NAME = "PythonDB"

    try:

        logger.info(
            "Starting database initialization process"
        )

        create_database(DATABASE_NAME)

        # IMPORTANT
        logger.info(
            "Waiting for database initialization..."
        )

        time.sleep(5)

        create_schemas(DATABASE_NAME)

        logger.info(
            "Database setup completed successfully"
        )

    except Exception:
        logger.exception(
            "Application execution failed"
        )

# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()