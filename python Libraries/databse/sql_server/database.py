import pyodbc 
import os 

def connect_to_database(database='master') :
    print("connecting to databse SQL Server....")

    try :
        conn = pyodbc.connect(
            "DRIVER={ODBC Driver 18 for SQL Server};"
            f"SERVER={os.getenv('SERVER')};"
            f"DATABASE={database};"
            f"UID={os.getenv('DB_USER')};"
            f"PWD={os.getenv('DB_PASSWORD')};"
            "Encrypt=yes;"
            "TrustServerCertificate=yes;"
        )
        print("Database connecting successfully :)")
        return conn

    except pyodbc.Error as e : 
        print(f"Error occurred during database connection : {e}")
        raise 
    
def create_database(conn) :
    print("dropping databse PythonDB if exist and recreate it")

    try :
        print("Dropping database PythonDB if exist")

        conn.autocommit = True
        cursor = conn.cursor()

        cursor.execute("""
                IF EXISTS(SELECT 1 FROM sys.databases WHERE name = 'PythonDB')
                BEGIN
                    ALTER DATABASE PythonDB SET SINGLE_USER WITH ROLLBACK IMMEDIATE ;
                    DROP DATABASE PythonDB ;
                END ;
        """)

        print("Creating database pythonDB.....")
        cursor.execute("CREATE DATABASE PythonDB;")

    except pyodbc.Error as e :
        print(f"Databse creating Faild {e}")
        raise

def main() :
    try :
        conn = connect_to_database()
        create_database(conn)
        conn.commit()
        conn.close()
    except :
        print("Error Occurred")


if __name__ == "__main__" :
    main()