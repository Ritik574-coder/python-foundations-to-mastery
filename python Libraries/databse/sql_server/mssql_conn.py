import os
import logging
from dotenv import load_dotenv
import pyodbc

# =========================
# Load Environment Variables
# =========================

load_dotenv()

DB_SERVER = os.getenv("DB_SERVER")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# =========================
# Validation
# =========================

if not DB_SERVER:
    raise ValueError("DB_SERVER not found in environment variables")

if not DB_USER:
    raise ValueError("DB_USER not found in environment variables")

if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD not found in environment variables")

# =========================
# Logging Configuration
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# =========================
# Database Connection
# =========================

def get_connection(database: str = "master") -> pyodbc.Connection:
    """
    Create and return SQL Server database connection.
    """

    try:
        connection_string = (
            "DRIVER={ODBC Driver 18 for SQL Server};"
            f"SERVER={DB_SERVER};"
            f"DATABASE={database};"
            f"UID={DB_USER};"
            f"PWD={DB_PASSWORD};"
            "Encrypt=yes;"
            "TrustServerCertificate=yes;"
        )

        logger.info(f"Connecting to database: {database}")

        conn = pyodbc.connect(connection_string)

        logger.info("Database connection established successfully")

        return conn

    except pyodbc.Error as e:
        logger.exception("Failed to connect to SQL Server")
        raise


# =========================
# Create Database
# =========================

def create_database(database_name: str = "PythonDB") -> None:
    """
    Drop and recreate database.
    """

    try:

        with get_connection("master") as conn:

            conn.autocommit = True

            with conn.cursor() as cursor:

                logger.info(f"Dropping database if exists: {database_name}")

                cursor.execute(f"""
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
                """)

                logger.info(f"Creating database: {database_name}")

                cursor.execute(f"""
                    CREATE DATABASE [{database_name}];
                """)

                logger.info(f"Database created successfully: {database_name}")

    except pyodbc.Error:
        logger.exception("Database creation failed")
        raise


# =========================
# Create Schemas
# =========================

def create_schemas(database_name: str = "PythonDB") -> None:
    """
    Create Medallion Architecture schemas.
    """

    schemas = ["bronze", "silver", "gold"]

    try:

        with get_connection(database_name) as conn:

            with conn.cursor() as cursor:

                for schema in schemas:

                    logger.info(f"Creating schema: {schema}")

                    cursor.execute(f"""
                        IF NOT EXISTS (
                            SELECT *
                            FROM sys.schemas
                            WHERE name = '{schema}'
                        )
                        BEGIN
                            EXEC('CREATE SCHEMA {schema} AUTHORIZATION dbo');
                        END;
                    """)

                conn.commit()

                logger.info("All schemas created successfully")

    except pyodbc.Error:
        logger.exception("Schema creation failed")
        raise


# =========================
# Main Execution
# =========================

def main():

    DATABASE_NAME = "PythonDB"

    try:

        logger.info("Starting database setup process")

        create_database(DATABASE_NAME)

        create_schemas(DATABASE_NAME)

        logger.info("Database setup completed successfully")

    except Exception:
        logger.exception("Application execution failed")


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    main()