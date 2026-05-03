import pyodbc 
import os

def connnect_to_databse(database="master") :

    print("connecting to SQL Server........")
    try :
        conn = pyodbc.connect(
            "DRIVER={ODBC Driver 18 for SQL Server};"
            "SERVER=localhost,1433;"
            f"DATABASE={database};"
            f"UID={os.getenv('DB_USER')};"
            f"PWD={os.getenv('DB_PASSWORD')};"
            "Encrypt=yes;"
            "TrustServerCertificate=yes;"

        )
        print("database connectiong successfully :) ")
        return conn
    
    except pyodbc.Error as e :
        print(f"Databse connectiong faild {e}")
        raise

def create_database(conn):
    try :

        conn.autocommit = True
        print(">> dropping databse if exists...")

        cursor = conn.cursor()
        cursor.execute("""
                IF EXISTS (SELECT name FROM sys.databases WHERE name = 'PythonDB')
                BEGIN
                    ALTER DATABASE PythonDB SET SINGLE_USER WITH ROLLBACK IMMEDIATE ;
                    DROP DATABASE PythonDB ;
                END ;
        """)

        print("creating databse PythonDB")

        cursor.execute("CREATE DATABASE PythonDB ;")

    except pyodbc.Error as e :
        print(f"Error Occurred while creating database {e}")
        raise

def create_schema():
    try :
        print("Creating schema based on the Medallion architecture")

        conn = connnect_to_databse("PythonDB")
        cursor = conn.cursor()

        print(">> creating bronze schema.....")
        cursor.execute("""
                IF NOT EXISTS(SELECT * FROM sys.schemas WHERE name = 'bronze')
                BEGIN 
                       EXEC('CREATE SCHEMA bronze AUTHORIZATION dbo') ;
                END ;
        """)

        print(">> creating silver schema.....")
        cursor.execute("""
                IF NOT EXISTS(SELECT * FROM sys.schemas WHERE name = 'silver')
                BEGIN 
                       EXEC('CREATE SCHEMA silver AUTHORIZATION dbo') ;
                END ;
        """)

        print(">> creating gold schema.....")
        cursor.execute("""
                IF NOT EXISTS(SELECT * FROM sys.schemas WHERE name = 'gold')
                BEGIN 
                       EXEC('CREATE SCHEMA gold AUTHORIZATION dbo') ;
                END ;
        """)

        conn.commit()
        conn.close()

    except pyodbc.Error as e :
        print(f"Error occurred during schema creating : {e}")
        raise

def main() :
    try :
        conn = connnect_to_databse()
        create_database(conn)

        conn.close()
        create_schema()

    except Exception as e :
        print(f"An error occurred during excution : {e}")

if __name__ == "__main__" :
    main()
