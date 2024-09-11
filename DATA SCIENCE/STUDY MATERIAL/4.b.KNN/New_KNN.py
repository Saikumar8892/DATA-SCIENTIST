'''CRISP-ML(Q):
    
Business Problem:
There are a lot of assumptions in the diagnosis pertaining to cancer. In a few cases radiologists, 
pathologists and oncologists go wrong in diagnosing whether tumor is benign (non-cancerous) or malignant (cancerous). 
Hence team of physicians want us to build an AI application which will predict with confidence the presence of cancer 
in a patient. This will serve as a compliment to the physicians.

Business Objective: Maximize Cancer Detection
Business Constraints: Minimize Treatment Cost & Maximize Patient Convenience

Success Criteria: 
Business success criteria: Increase the correct diagnosis of cancer in at least 96% of patients
Machine Learning success criteria: Achieve an accuracy of atleast 98%
Economic success criteria: Reducing medical expenses will improve trust of patients and thereby hospital will see an increase in revenue by atleast 12%

Data Collection:
Data is collected from the hospital for 569 patients. 30 features and 1 label comprise the feature set. 
Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)'''
    
    
# CODE MODULARITY IS EXTREMELY IMPORTANT
# Import the libraries
# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For data visualization

# Import necessary modules from scikit-learn
from sklearn.impute import SimpleImputer  # For imputing missing values
from sklearn.preprocessing import OneHotEncoder  # For one-hot encoding categorical variables
from sklearn.preprocessing import MinMaxScaler  # For scaling numerical features

# Additional imports from scikit-learn
from sklearn.compose import ColumnTransformer  # For applying transformations to specific columns

# Import functions for model evaluation and selection
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors classifier
from sklearn.pipeline import Pipeline  # For constructing a pipeline of transformers and estimator

import sklearn.metrics as skmet  # For evaluating model performance
import pickle  # For saving the trained model to a file

# MySQL Database connection

# Importing necessary module for creating a database engine
from sqlalchemy import create_engine
from urllib.parse import quote
# Reading the cancer data from a CSV file into a pandas DataFrame
cancerdata = pd.read_csv(r"cancerdata.csv")

# Setting up connection parameters for the MySQL database
user = 'root'  # Username
pw = quote('Sai@123kumar')  # Password
db = 'knn'  # Database name

# Creating an engine to connect to the MySQL database using SQLAlchemy
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Dumping the cancer data into the MySQL database table named 'cancer'
# 'if_exists' parameter is set to 'replace' to replace the table if it already exists
# 'chunksize' parameter is used to specify the number of rows to write at a time
# 'index' parameter is set to False to avoid writing DataFrame index as a column in the table
cancerdata.to_sql('cancer', con=engine, if_exists='replace', chunksize=1000, index=False)

# loading data from database
# SQL query to select all records from the 'cancer' table in the MySQL database
sql = 'select * from cancer'

# Reading data from the MySQL database table 'cancer' into a pandas DataFrame
cancerdf = pd.read_sql_query(sql, con=engine)

# Displaying the DataFrame
print(cancerdf)

# Data Preprocessing & Exploratory Data Analysis (EDA)

# Converting 'B' to 'Benign' and 'M' to 'Malignant' in the 'diagnosis' column
cancerdf['diagnosis'] = np.where(cancerdf['diagnosis'] == 'B', 'Benign', cancerdf['diagnosis'])
cancerdf['diagnosis'] = np.where(cancerdf['diagnosis'] == 'M', 'Malignant', cancerdf['diagnosis'])

# Dropping the 'id' column from the DataFrame
cancerdf.drop(['id'], axis=1, inplace=True)

# Displaying information about the DataFrame, including the data types and non-null values
cancerdf.info()

# Generating descriptive statistics of the DataFrame, including count, mean, std, min, max, etc.
cancerdf.describe()

# Creating a new DataFrame 'cancerdf_X' containing all columns except the first column ('diagnosis')
cancerdf_X = pd.DataFrame(cancerdf.iloc[:, 1:])

# Creating a new DataFrame 'cancerdf_y' containing only the first column ('diagnosis')
cancerdf_y = pd.DataFrame(cancerdf.iloc[:, 0])

# Displaying information about the DataFrame 'cancerdf_X', including data types and non-null counts
cancerdf_X.info()

# Selecting numerical features from the DataFrame 'cancerdf_X'
numeric_features = cancerdf_X.select_dtypes(exclude=['object']).columns

# Displaying the names of the numerical features
numeric_features

# Constructing a pipeline for numerical feature processing
# Define imputation and scaling pipeline
num_pipeline = Pipeline([('impute', SimpleImputer(strategy='mean')), ('scale', MinMaxScaler())])

# Selecting categorical features from the DataFrame 'cancerdf_X'
categorical_features = cancerdf_X.select_dtypes(include=['object']).columns

# Displaying the names of the categorical features
categorical_features


# Constructing a pipeline for categorical feature processing using DataFrameMapper and OneHotEncoder
categ_pipeline = Pipeline([('encoding', OneHotEncoder(drop='first'))])# Encoding categorical features using OneHotEncoder

# Constructing a preprocessing pipeline using ColumnTransformer
# This pipeline applies the categorical pipeline to categorical features and the numerical pipeline to numerical features
preprocess_pipeline = ColumnTransformer([
    ('categorical', categ_pipeline, categorical_features),  # Applying categorical pipeline to categorical features
    ('numerical', num_pipeline, numeric_features)  # Applying numerical pipeline to numerical features
])

# Fitting the preprocessing pipeline to the data and transforming the data
processed = preprocess_pipeline.fit(cancerdf_X)

# Displaying the processed data
processed

# Importing joblib module for saving the processed pipeline
import joblib

# Saving the processed pipeline to a file named 'processed1' using joblib
joblib.dump(processed, 'processed1')

# Importing the os module to get the current working directory
import os 

# Getting the current working directory
os.getcwd()

# Transforming the original data using the preprocessing pipeline defined above
# Creating a new DataFrame 'cancerclean' with the cleaned and processed data for clustering
cancerclean = pd.DataFrame(processed.transform(cancerdf_X), columns=processed.get_feature_names_out())

# Displaying information about the DataFrame 'cancerclean', including data types and non-null counts
cancerclean.info()

# Selecting new features (columns) from the cleaned DataFrame 'cancerclean' excluding columns with data type 'object'
# new_features = cancerclean.select_dtypes(exclude=['object']).columns 
# new_features

# Generating descriptive statistics of the transformed DataFrame 'cancerclean_n'
res = cancerclean.describe()
res

# Separating predictors (X) and target (Y) from the transformed data for modeling
# X = np.array(cancerclean_n.iloc[:, :])  # Predictors
Y = np.array(cancerdf_y['diagnosis'])  # Target

# Splitting the transformed data into train and test sets
# 80% of the data will be used for training and 20% for testing
# Setting a random seed for reproducibility
X_train, X_test, Y_train, Y_test = train_test_split(cancerclean, Y,
                                                    test_size=0.2, random_state=0)

# Displaying the shapes of the train and test sets
X_train.shape
X_test.shape

# Creating a kNN classifier with 21 neighbors
knn = KNeighborsClassifier(n_neighbors=21)

# Training the kNN model on the training data
KNN = knn.fit(X_train, Y_train)

# Predicting the classes on the training data using the trained model
pred_train = knn.predict(X_train)

# Displaying the predicted classes on the training data
pred_train

# Creating a confusion matrix to compare actual and predicted classes on the training data
# Rows represent actual classes and columns represent predicted classes
# Printing the confusion matrix with appropriate row and column names
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames=['Predictions']) 

# Calculating and printing the accuracy of the model on the training data
print(skmet.accuracy_score(Y_train, pred_train))

# Predicting the classes on the test data using the trained model
pred = knn.predict(X_test)

# Displaying the predicted classes on the test data
pred

# Printing the accuracy of the model on the test data
print(skmet.accuracy_score(Y_test, pred))

# Creating a confusion matrix to compare actual and predicted classes on the test data
# Rows represent actual classes and columns represent predicted classes
# Printing the confusion matrix with appropriate row and column names
pd.crosstab(Y_test, pred, rownames=['Actual'], colnames=['Predictions']) 

# Calculating the confusion matrix for the test data
cm = skmet.confusion_matrix(Y_test, pred)

# Creating a ConfusionMatrixDisplay object to plot the confusion matrix
# Displaying the confusion matrix with appropriate labels and title
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
cmplot.plot()
cmplot.ax_.set(title='Cancer Detection - Confusion Matrix', xlabel='Predicted Value', ylabel='Actual Value')

# Initializing an empty list to store accuracy values
acc = []

# Looping through a range of odd numbers from 3 to 49 with a step size of 2
# This loop iterates over different values of k (number of neighbors) for kNN classifier
for i in range(3, 50, 2):
    # Creating a kNN classifier with i neighbors
    neigh = KNeighborsClassifier(n_neighbors=i)
    
    # Training the kNN classifier on the training data
    neigh.fit(X_train, Y_train)
    
    # Calculating the training accuracy by comparing predicted labels with actual labels for the training data
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    
    # Calculating the test accuracy by comparing predicted labels with actual labels for the test data
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    
    # Calculating the difference between training and test accuracies
    diff = train_acc - test_acc
    
    # Appending the difference, training accuracy, and test accuracy to the acc list
    acc.append([diff, train_acc, test_acc])

# Displaying the acc list, which contains differences, training accuracies, and test accuracies for each k value
acc

# Plotting the training and test accuracies for different values of k
# Red circles represent training accuracies
plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], "ro-")

# Blue circles represent test accuracies
plt.plot(np.arange(3, 50, 2), [i[2] for i in acc], "bo-")


# Importing GridSearchCV from sklearn.model_selection
from sklearn.model_selection import GridSearchCV

# Creating a list of k values from 3 to 49 with a step size of 2
k_range = list(range(3, 50, 2))

# Creating a dictionary containing the parameter grid for GridSearchCV
param_grid = dict(n_neighbors=k_range)

# Creating a GridSearchCV object named 'grid'
# cv=5 specifies 5-fold cross-validation
# scoring='accuracy' indicates using accuracy as the evaluation metric
# return_train_score=False means not to return the training scores
# verbose=1 prints the progress messages
grid = GridSearchCV(knn, param_grid, cv=5, 
                    scoring='accuracy', 
                    return_train_score=False, verbose=1)

# Printing the documentation/help for the GridSearchCV class
help(GridSearchCV)

# Fitting the GridSearchCV object 'grid' on the training data (X_train, Y_train)
KNN_new = grid.fit(X_train, Y_train) 

# Printing the best parameters found by GridSearchCV
print(KNN_new.best_params_)

# Calculating the accuracy score for the best model found by GridSearchCV
accuracy = KNN_new.best_score_ * 100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )

# Predicting the classes on the test data using the best model found by GridSearchCV
pred = KNN_new.predict(X_test)
pred

# Calculating the confusion matrix for the test data using the best model
cm = skmet.confusion_matrix(Y_test, pred)

# Creating a ConfusionMatrixDisplay object to plot the confusion matrix
# Displaying the confusion matrix with appropriate labels and title
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
cmplot.plot()
cmplot.ax_.set(title='Cancer Detection - Confusion Matrix', xlabel='Predicted Value', ylabel='Actual Value')

# Retrieving the best kNN classifier from the GridSearchCV object
knn_best = KNN_new.best_estimator_

# Saving the best kNN classifier to a file named 'knn.pkl' using pickle
pickle.dump(knn_best, open('knn.pkl', 'wb'))

# Getting the current working directory
import os
os.getcwd()
