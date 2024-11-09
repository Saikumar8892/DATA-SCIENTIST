import pandas as pd  # Import the pandas library for data manipulation
import numpy as np  # Import the numpy library for numerical computations
import matplotlib.pyplot as plt  # Import the matplotlib library for data visualization

from sklearn.compose import ColumnTransformer  # Import ColumnTransformer for preprocessing pipelines
from sklearn.pipeline import Pipeline  # Import Pipeline for chaining preprocessing steps
from sklearn.impute import SimpleImputer  # Import SimpleImputer for missing value imputation
from feature_engine.outliers import Winsorizer  # Import Winsorizer for outlier treatment
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler for feature scaling

from sklearn.model_selection import train_test_split  # Import train_test_split for splitting data into train and test sets
from sklearn.svm import SVC  # Import SVC (Support Vector Classifier) from sklearn for SVM classification
from sklearn.model_selection import RandomizedSearchCV  # Import RandomizedSearchCV for hyperparameter tuning
import pickle, joblib  # Import pickle and joblib for model serialization
from getpass import getpass
from sqlalchemy import create_engine  # Import create_engine from SQLAlchemy for database connection
from urllib.parse import quote

user = 'root'  # user name
db = 'svm'  # database name
your_password = getpass()
engine = create_engine(f"mysql+pymysql://{user}:{your_password}@localhost/{db}")
# Create a database connection engine using SQLAlchemy.
# 'create_engine' creates a connection to the MySQL database specified by the user, password, and database name.
# 'f"mysql+pymysql://{user}:{pw}@localhost/{db}"' constructs the connection string with the provided user credentials and database name.
# 'localhost' indicates that the database is hosted locally.

# Load the dataset from the uploaded CSV file
data = pd.read_csv('SalaryData_Train.csv')

data.to_sql('data', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
# Load the dataset from the uploaded CSV file
data = pd.read_csv('SalaryData_Train.csv')

# Display basic information about the dataset and the first few rows
data_info = data.info()
data_head = data.head()

data_info, data_head

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if column != 'Salary':  # Exclude target variable
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Encode the target variable
data['Salary'] = LabelEncoder().fit_transform(data['Salary'])

# Scale numerical features
scaler = StandardScaler()
data[['age', 'educationno', 'capitalgain', 'capitalloss', 'hoursperweek']] = scaler.fit_transform(
    data[['age', 'educationno', 'capitalgain', 'capitalloss', 'hoursperweek']])

# Split data into features (X) and target (Y)
X = data.drop(columns='Salary')
Y = data['Salary']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Train an SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

accuracy