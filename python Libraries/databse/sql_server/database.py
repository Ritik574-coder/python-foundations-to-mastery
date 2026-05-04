import pyodbc 
import os

def connect_to_database() :
    print("connecting to SQL Server databse....")

    try :
        conn = pyodbc.connect(
            "DRIVER={ODBC Driver 18 for SQL Server};"
            "SERVER=localhost,1433;"
            "DATABASE=master;"
            f"UID={os.getenv('DB_USER')};"
            f"PWD={os.getenv('DB_PASSWORD')};"
            "Encrypt=yes;"
            "TrustServerCertificate=yes;"
        )
        
        print("Databse connection successfully :) ")
        return conn 
    
    except pyodbc.Error as e :
        print(f"Database connection faild : {e}")
        raise 

def create_database(conn) :
    try :
        conn.autocommit = True
    except pyodbc.Error as e :
        print("Error occurred during database creation")