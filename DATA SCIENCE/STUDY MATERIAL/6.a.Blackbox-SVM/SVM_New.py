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

from sqlalchemy import create_engine  # Import create_engine from SQLAlchemy for database connection
from urllib.parse import quote

# Database connection details
user = 'root'  # user name
pw = quote('Sai@123kumar' ) # password
db = 'cars'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
# Create a database connection engine using SQLAlchemy.
# 'create_engine' creates a connection to the MySQL database specified by the user, password, and database name.
# 'f"mysql+pymysql://{user}:{pw}@localhost/{db}"' constructs the connection string with the provided user credentials and database name.
# 'localhost' indicates that the database is hosted locally.
letters = pd.read_csv(r"letterdata.csv")

letters.to_sql('letters_svm', con = engine, if_exists = 'replace', chunksize = 1000, index = False)



# SQL query to fetch data from the database
sql = 'select * from letters_svm;'
# Define an SQL query to select all columns and rows from the 'letters_svm' table.

# Read data from SQL database into a DataFrame
letters = pd.read_sql_query(sql, engine)
# Use 'pd.read_sql_query' to execute the SQL query and read the result into a pandas DataFrame.
# 'sql' is the SQL query string, and 'engine' is the SQLAlchemy engine object created earlier.

# Display summary statistics of the DataFrame
letters.describe()
# Use the 'describe' method to display summary statistics (count, mean, std, min, 25%, 50%, 75%, max) of the DataFrame.
# This provides an overview of the numerical columns in the DataFrame.

# Predictors and Target variables
X = letters.iloc[:, 1:]  # Extract features (predictors) from the DataFrame 'letters', excluding the first column (target variable)
Y = letters.iloc[:, 0]   # Extract the target variable from the first column of the DataFrame 'letters'

# Numeric features in the dataset
numeric_features = X.select_dtypes(exclude=['object']).columns
# Select numeric features (columns with data type other than 'object') from the DataFrame 'X'
# and store their column names in the variable 'numeric_features'

# Information about numeric features
X[numeric_features].info()
# Display information about the numeric features in the DataFrame 'X', including data type and non-null count
X[numeric_features].info()
# Pipeline for numerical data preprocessing
num_pipeline = Pipeline([('impute', SimpleImputer(strategy = 'mean')),# Create a pipeline for numerical data preprocessing, which includes imputation of missing values using mean strategy.
                         ('winsorize', Winsorizer(capping_method = 'iqr', tail='both', fold=1.5)),# Create a pipeline for the Winsorizer
                         ('scale', MinMaxScaler())])# Create a pipeline for MinMaxScaler



# Imputation Transformer
preprocessor = ColumnTransformer([('clean', num_pipeline, numeric_features)])
print(preprocessor)

clean_data = preprocessor.fit(X)

# Save the data preprocessing pipeline
joblib.dump(clean_data, 'clean')

X1 = pd.DataFrame(clean_data.transform(X), columns = X.columns)
X1# Get summary statistics of the scaled data

# Split the data into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(X1, Y, test_size=0.2, stratify=Y)
#  - train_X: Training features
#  - test_X: Testing features
#  - train_y: Training labels (target variable)
#  - test_y: Testing labels
#  - test_size=0.2: Split 20% of the data for testing
#  - stratify=Y: Stratified split to maintain class proportions in training and testing sets (assuming Y is the categorical target variable)

# Support Vector Classifier with linear kernel
model_linear = SVC(kernel="linear")  # Create a model with linear kernel
model1 = model_linear.fit(train_X, train_y)  # Train the model on the training data
pred_test_linear = model_linear.predict(test_X)  # Make predictions on the testing data


# Accuracy of the model with linear kernel
np.mean(pred_test_linear == test_y)  # Calculate accuracy (percentage of correct predictions)

# Hyperparameter optimization using RandomizedSearchCV
model = SVC()  # Define the base model (Support Vector Classifier)

# Hyperparameter grid for searching
parameters = {'C': [0.1, 1, 10, 100],  # Regularization parameter
              'gamma': [1, 0.1, 0.01, 0.001],  # Kernel coefficient for some kernels
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}  # Kernel function types

rand_search = RandomizedSearchCV(model, parameters, n_iter=10, n_jobs=3, cv=3, scoring='accuracy', random_state=0)
#  - model: The base model (SVC)
#  - parameters: Dictionary of hyperparameters to search over
#  - n_iter=10: Number of random parameter combinations to try
#  - n_jobs=3: Use 3 cores for parallel execution (if available)
#  - cv=3: Use 3-fold cross-validation for evaluation during search
#  - scoring='accuracy': Use accuracy as the evaluation metric
#  - random_state=0: Set random seed for reproducibility

randomised = rand_search.fit(train_X, train_y)  # Perform randomized hyperparameter search

# Best parameters found by RandomizedSearchCV
randomised.best_params_  # Print the best hyperparameter combination identified

# Best model (estimator) from the randomized search
best = randomised.best_estimator_  # Access the model with the best hyperparameters

# Evaluate the best model on the testing data
pred_test = best.predict(test_X)  # Make predictions on the testing data using the best model
accuracy = np.mean(pred_test == test_y)  # Calculate accuracy of the best model

# Print or store the accuracy of the best model (e.g., print(f"Accuracy of best model: {accuracy:.4f}"))

# Save the best model for future use
pickle.dump(best, open('svc_rcv.pkl', 'wb'))  # Save the best model using pickle

# Load new data for prediction (assuming 'test_svm.csv' is in the specified path)
data = pd.read_csv(r"test_svm.csv")

# Select numeric features from the new data (assuming some features might be categorical)
numeric_features = data.select_dtypes(exclude=['object']).columns  # Get columns with numerical data types

# Load the saved model and preprocessing transformers

  
model1 = pickle.load(open('svc_rcv.pkl', 'rb'))
clean = joblib.load('clean')

# Apply preprocessing steps sequentially to the new data
clean = pd.DataFrame(clean.transform(data), columns=data.columns)  

# Make predictions on the preprocessed data using the loaded model
prediction = pd.DataFrame(model1.predict(clean), columns=['choice_pred'])  # Create DataFrame with predictions

# Combine predictions with the original data for easier analysis
final = pd.concat([prediction, data], axis=1)  # Concatenate prediction DataFrame with original data
