import pyodbc
import os 

def connect_to_database(database='master') :
    print("connecting to sql server")

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

        print("Database connecting successfully :) ")
        return conn

    except pyodbc.Error as e : 
        print(f"databse connectin faild {e}")
        raise 

def create_database() :
    print("creating database for preastic")
