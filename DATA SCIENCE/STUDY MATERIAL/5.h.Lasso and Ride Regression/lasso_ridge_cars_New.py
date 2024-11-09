## Problem Statement
'''
The client/business deals with used cars sales.

The customers in this sector give strong preference to less-aged cars 
and popular brands with good resale value. 
This puts a very strong challenge as they only have a very limited range of 
vehicle options to showcase.

#### No Pre-Set Standards

`How does one determine the value of a used car?`

The Market scenario is filled with a lot of malpractices. There is no defined standards exist to determine the appropriate price for the cars, the values are determined by arbitrary methods.

The unorganized and unstructured methods are disadvantageous to the both the parties trying to strike a deal. The look and feel can be altered in used cars, but the performance cannot be altered beyond a point.


#### Revolutionizing the Used Car Industry Through Machine Learning

**Linear regression**
Linear regression is a ML model that estimates the relationship between independent variables and a dependent variable using a linear equation (straight line equation) in a multidimensional space.

**CRISP-ML(Q) process model describes six phases:**

- Business and Data Understanding
- Data Preparation (Data Engineering)
- Model Building (Machine Learning)
- Model Evaluation and Tunning
- Deployment
- Monitoring and Maintenance


**Objective(s):** Maximize the profits

**Constraints:** Maximize the customer satisfaction

**Success Criteria**

- **Business Success Criteria**: Improve the profits from anywhere between 10% to 20%

- **ML Success Criteria**: RMSE should be less than 0.15

- **Economic Success Criteria**: Second/Used cars sales delars would see an 
increase in revenues by atleast 20%
'''

# Load the Data and perform EDA and Data Preprocessing

# Import necessary libraries

# Import libraries for data manipulation and analysis
import pandas as pd  # for dataframes and data analysis
import numpy as np  # for numerical operations and arrays

# Import libraries for data visualization
import matplotlib.pyplot as plt  # for creating plots and charts

#pip install sidetable
# Import library for printing fancy data tables (optional)
import sidetable  # for creating visually appealing data tables

# Import libraries for building and evaluating machine learning models
import statsmodels.api as sm  # Library for statistical modeling and testing
from sklearn.linear_model import Ridge, Lasso, ElasticNet  # Models for Ridge, Lasso, and ElasticNet regression
from sklearn.compose import ColumnTransformer  # for combining preprocessing steps
from sklearn.pipeline import Pipeline  # for chaining data processing steps
from sklearn.impute import SimpleImputer  # for handling missing data
from sklearn.preprocessing import MinMaxScaler  # for scaling numerical features
from sklearn.preprocessing import OneHotEncoder  # for encoding categorical features
from feature_engine.outliers import Winsorizer  # for capping outliers (optional)

# Import libraries for statistical analysis
from statsmodels.tools.tools import add_constant  # for adding a constant term to models
from statsmodels.stats.outliers_influence import variance_inflation_factor  # for detecting multicollinearity

# Import libraries for model persistence
import joblib  # for saving and loading scikit-learn models (recommended)
import pickle  # for saving and loading Python objects (less secure than joblib)

# Import library for hyperparameter tuning (optional)
from sklearn.model_selection import GridSearchCV  # for grid search cross-validation

# Import library for database connection (optional)
from sqlalchemy import create_engine  # for connecting to databases
from urllib.parse import quote
# Create a connection engine to the MySQL database named 'cars_db'
# using credentials 'user1' and 'user1' (replace with your actual credentials)
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                        .format(user="root", pw=quote("Sai@123kumar"), db="cars"))
pip install sqlalchemy pymysql

# Read the 'cars' table from the MySQL database into a Pandas DataFrame
sql = 'SELECT * FROM cars'
dataset = pd.read_sql_query(sql, engine)

# Display summary statistics of the dataset's numerical columns
dataset.describe()

# Check for missing values in each column (True indicates missing values)
print(dataset.isnull().any())

# Display data types and non-null counts for each column
dataset.info()

# Separate the feature matrix (X) containing all columns except the first
# (assuming the first column is the target variable)
X = dataset.iloc[:, 1:6]  # Select all rows and columns from 1 (inclusive) to 6 (exclusive)

# Create the target variable DataFrame (y) containing the first column
y = dataset.iloc[:, 0]  # Select all rows and the first column (0)


# Explore unique values and counts for the categorical feature "Enginetype"
print(X["Enginetype"].unique())  # Display unique engine types
print(X["Enginetype"].value_counts())  # Count occurrences of each engine type

# Explore frequencies of all categorical variables using sidetable (optional)
X.stb.freq(["Enginetype"])  # Display frequencies of all categorical features (if sidetable is imported)

# Separate categorical and numerical features based on data types
categorical_features = X.select_dtypes(include=['object']).columns  # Get categorical column names
numeric_features = X.select_dtypes(exclude=['object']).columns     # Get numerical column names

## Missing values Analysis
# Define pipeline for missing data if any
num_pipeline = Pipeline([('impute', SimpleImputer(strategy = 'mean')),
                         ('scale', MinMaxScaler()),
                         ('winsorize', Winsorizer(capping_method = 'iqr', tail='both', fold=1.5)),
                         ])

# Encoding categorical to numeric variable
categ_pipeline = Pipeline([('label', OneHotEncoder())])

# Using ColumnTransfer to transform the columns of an array or pandas DataFrame. This estimator allows different columns or column subsets of the input to be transformed separately and the features generated by each transformer will be concatenated to form a single feature space.
preprocess_pipeline = ColumnTransformer([('numerical', num_pipeline, numeric_features), 
                                         ('categorical', categ_pipeline, categorical_features)])

clean = preprocess_pipeline.fit(X)  # Pass the raw data through pipeline
clean # imputation scalaing winsorisation encoding 

# Save the defined pipeline
joblib.dump(clean, 'clean') 

import os 
os.getcwd()

# Transform the original data using the pipeline defined above
clean_data = pd.DataFrame(clean.transform(X), columns = clean.get_feature_names_out()) 



# Create a constant term (usually for intercept) and add it to the cleaned data
# for use with statsmodels' OLS model
P = add_constant(clean_data)

# Fit a statsmodels Ordinary Least Squares (OLS) model on the combined data (P) and target variable (y)
basemodel = sm.OLS(y, P).fit()  # 'sm' likely refers to 'statsmodels'

# Check for multicollinearity using Variance Inflation Factor (VIF)
vif = pd.Series([variance_inflation_factor(P.values, i) for i in range(P.shape[1])],
                index=P.columns)  # Calculate VIF for each feature

# Identify features with high VIF (potential multicollinearity)
# You may need to adjust the threshold based on your domain knowledge and tolerance
high_vif_cols = vif[vif > 5]  # Example: Consider features with VIF > 5 to be highly correlated
print("Features with potentially high multicollinearity (VIF > 5):", high_vif_cols.index.tolist())

# Create a new DataFrame 'clean_data1' by dropping a feature with high VIF (optional)
# Consider using domain knowledge and feature importance along with VIF to decide which feature to remove
clean_data1 = P.drop('numerical__WT', axis=1)  # Example: Drop the 'WT' feature (replace with the actual high VIF feature)

# Refit the OLS model on the data without the potentially problematic feature
basemodel2 = sm.OLS(y, clean_data1).fit()


# Hyperparameter tuning for Lasso, Ridge, and ElasticNet regression
lasso = Lasso()  # Initialize Lasso regression model

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.13, 1, 5, 10, 20]}  # Define alpha values for hyperparameter tuning

lasso_reg = GridSearchCV(lasso, parameters, scoring='r2', cv=5)  # Perform grid search with cross-validation to find best alpha for Lasso

lasso_reg.fit(clean_data1, y)  # Fit Lasso regression model to the cleaned data

lasso_pred = lasso_reg.predict(clean_data1)  # Make predictions using the trained Lasso regression model



ridge = Ridge()  # Initialize Ridge regression model

ridge_reg = GridSearchCV(ridge, parameters, scoring='r2', cv=5)  # Perform grid search with cross-validation to find best alpha for Ridge

ridge_reg.fit(clean_data1, y)  # Fit Ridge regression model to the cleaned data

ridge_pred = ridge_reg.predict(clean_data1)  # Make predictions using the trained Ridge regression model



enet = ElasticNet()  # Initialize ElasticNet regression model

enet_reg = GridSearchCV(enet, parameters, scoring='r2', cv=5)  # Perform grid search with cross-validation to find best alpha for ElasticNet

enet_reg.fit(clean_data1, y)  # Fit ElasticNet regression model to the cleaned data

enet_pred = enet_reg.predict(clean_data1)  # Make predictions using the trained ElasticNet regression model


# Compare scores of different regression models
scores_all = pd.DataFrame({'models': ['Lasso', 'Ridge', 'Elasticnet', 'Grid_lasso', 'Grid_ridge', 'Grid_elasticnet'],
                           'Scores': [lasso_reg.best_score_, ridge_reg.best_score_, enet_reg.best_score_,
                                      lasso_reg.best_score_, ridge_reg.best_score_, enet_reg.best_score_]})

# Save the best model
# Extract the best ElasticNet model from the GridSearchCV object
finalgrid = enet_reg.best_estimator_

# Save the best ElasticNet model for later use
pickle.dump(finalgrid, open('grid_elasticnet.pkl', 'wb'))  # 'wb' for binary write

# Load the best ElasticNet model for prediction
model1 = pickle.load(open('grid_elasticnet.pkl', 'rb'))  # 'rb' for binary read

# Load the previously saved preprocessing models
clean = joblib.load('clean')


# Read the test data from an Excel file
data = pd.read_excel(r"carswithenginetype_test.xlsx")

# Preprocess the test data
clean_data = pd.DataFrame(clean.transform(data), columns = clean.get_feature_names_out())
P = add_constant(clean_data)
clean_data1 = P.drop('numerical__WT', axis = 1)
# Make predictions on the preprocessed test data using the best ElasticNet model
prediction = pd.DataFrame(model1.predict(clean_data1), columns=['Predict_MPG'])

# Combine predictions with the original test data for analysis
final = pd.concat([prediction, data], axis=1)

# Display the final DataFrame containing predictions and original data
print(final)

