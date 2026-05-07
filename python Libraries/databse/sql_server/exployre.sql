-- creating database 
IF EXISTS(SELECT 1 FROM sys.databases WHERE name = 'PythonDB')
BEGIN 
    ALTER DATABASE PythonDB SET SINGLE_USER WITH ROLLBACK IMMEDIATE ;
    DROP DATABASE PythonDB ;
END ;

-- creatin bronze schema 
IF NOT EXISTS(SELECT 1 FROM sys.schemas WHERE name = 'bronze')
BEGIN
    EXEC('CREATE SCHEMA bronze AUTHORIZATION dbo') ;
END ;

-- creatin silver scheam 
IF NOT EXISTS(SELECT 1 FROM sys.schemas WHERE name = 'bronze')
BEGIN
    EXEC('CREATE SCHEMA bronze AUTHORIZATION dbo') ;
END ;

-- creating gold schema 
IF NOT EXISTS(SELECT 1 FROM sys.schemas WHERE name = 'bronze')
BEGIN
    EXEC('CREATE SCHEMA bronze AUTHORIZATION dbo') ;
END ;