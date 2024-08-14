# import libraries
import pandas as pd
# Import data (.csv file) using pandas. We are using mba data set
mba = pd.read_csv(r"education.csv")
mba.info()

# - SQLAlchemy is basically referred to as the toolkit of Python SQL that 
# provides developers with the flexibility of using the SQL database. 
# - The benefit of using this particular library is to allow Python developers 
# to work with the language's own objects, and not write separate SQL queries.

pip install sqlalchemy
from sqlalchemy import create_engine, text

# **Engine Configuration**
# The Engine is the starting point for any SQLAlchemy application. 
# It’s “home base” for the actual database and its DBAPI, 
# delivered to the SQLAlchemy application through a connection pool and a Dialect, 
# which describes how to talk to a specific kind of database/DBAPI combination.
# sqlalchemy helps to connect mysql, postgresql, microsoftsql(mssql), etc;
# Refer this url for brief Knowledge about "create_engine"
# https://docs.sqlalchemy.org/en/20/core/engines.html
pip install urllib3
from urllib.parse import quote
# Required to handle the differences between SQL and Python 
# SQL is case insenstive
# SQL will allow special characters

# The quote function in urllib.parse takes a string as input and returns 
# a URL-encoded version of that string. 
# It replaces special characters with their corresponding percent-encoded representation, 
# which consists of a '%' sign followed by two hexadecimal digits.

## For mysql
pip install pymysql
# Creating engine which connect to MySQL
user = 'root' # user name
pw = 'Sai@123kumar' # password
db = '360db' # database

# creating engine to connect database
# If any special characters are there in sql username and password
engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))

# if sql username and password do not have any special characters or upper case letters
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# To send data into DataBase
# DataFrame.to_sql(name, con, schema=None, if_exists='fail', index=True, index_label=None, chunksize=None, dtype=None, method=None)
# Go to this link for more details
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html

# Tables can be newly created, appended to, or overwritten.
mba.to_sql('mba', con = engine, if_exists = 'replace', chunksize = None, index= False)
# sending data into database and connecting with Engine by using "DataFrame.to_sql()"

# To get the data From DataBase
sql = "SELECT * FROM education;" # wright query of sql and save into variable
sql = "SELECT * FROM student_detail;" 
# Note: If sqlalchmey version 1.4.x
df1 = pd.read_sql_query(sql, engine) # connecting query with Engine and reading the results by using "pd.read_sql_query"


# Note: If sqlalchmey version is 2.x
df1 = pd.read_sql_query(text(sql), engine.connect()) # connecting query with Engine and reading the results by using "pd.read_sql_query"
