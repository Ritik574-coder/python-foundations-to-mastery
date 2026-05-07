import pyodbc 
import os 

def connect_to_database(database="master") :
    print(">> Connecting to SQL Server datasbe... ")

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

        print(">> Database connecting successfully...")
        return conn

    except pyodbc.Error as e : 
        print(f"Error occurred during database connction : {e}")
        raise 

def creatign_databse(conn) :
    print(">> Creatin database PythonDB if exists drop database and recreate..")

    try : 
        conn.autocommit = True 
        print(">> Dropping Databse if exists...")
        cursor = conn.cursor()

        cursor.execute("""
        IF EXISTS(SELECT 1 FROM sys.databases WHERE name = 'PythonDB')
        BEGIN 
            ALTER DATABASE PythonDB SET SINGLE_USER WITH ROLLBACK IMMEDIATE ;
            DROP DATABASE PythonDB ;
        END ;
        """)

    except pyodbc.Error as e :
        print(f"Error occurred during database creation {e}")
        raise 