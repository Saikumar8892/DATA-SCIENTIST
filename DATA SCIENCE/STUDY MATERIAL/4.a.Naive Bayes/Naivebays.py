# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:01:02 2024

@author: DELL
"""

'''CRISP-ML(Q)

a. Business & Data Understanding
    As internet penetration is increasing the usage of electronic media as a 
    mode of effective communication is increasing.     So are the spamsters
    who master in spamming your mailbox with innovative emails, which are
    difficult to classify as spam. A few of these might also have viruses and 
    might trick you into losing money via fraud black hat techniques. 
    The same logic applies to Telecom companies when it comes to SMS - Short Messaging Service.

    i. Business Objective - Maximize Spam Detection
    ii. Business Constraint - Minimize Manual Spam Detection Rules

    Success Criteria:
    1. Business Success Criteria - Reduce the customer churn by 12%.
    2. ML Success Criteria - Achieve an accuracy of over 80% & performance of
    detecting span for streaming data.
    3. Economic Success Criteria - The cost of acquiring new customers is 
    10 times costlier than retaining an existing customer. Hence by reducing 
    customer churn by 12%, one can get a cost savings of approximately 120K USD to 130K USD.
    Note: These numbers are hypothetical
    
    
    Data Collection - SMS spam collection data from Telecom company is obtained
    where labels were manually given by the employees of the company.
    
    Data has 5559 observations and 2 columns. 
    
    Metadata Description:
    Column Name = Type - The output variable and has '2' classes - spam & ham
    Column Name = Text - The input variable and contains the SMS received by customers'''

# Code modularity must be maintained
# Import all the required libraries and modules

# Importing the pandas module as pd
import pandas as pd
# Importing the numpy module as np
import numpy as np
# Importing the train_test_split function from the sklearn.model_selection module
from sklearn.model_selection import train_test_split

# Imbalanced-learn pipeline is being called using SMOTE in our pipeline.
# pip install imblearn
# Importing the make_pipeline function from the imblearn.pipeline module
from imblearn.pipeline import make_pipeline
# Importing the SMOTE class from the imblearn.over_sampling module
from imblearn.over_sampling import SMOTE

# Importing the MultinomialNB class from the sklearn.naive_bayes module
from sklearn.naive_bayes import MultinomialNB

# Importing the CountVectorizer class from the sklearn.feature_extraction.text module
from sklearn.feature_extraction.text import CountVectorizer
# Importing the metrics submodule from the sklearn.metrics module as skmet
import sklearn.metrics as skmet

# Importing the joblib module for saving and loading models
import joblib

# Importing the GridSearchCV class from the sklearn.model_selection module

from sklearn.model_selection import GridSearchCV

# Loading the data set
# Reading the CSV file into a DataFrame 'data'
# Parameters:
#   - "sms_raw_NB.csv": File path of the CSV file to be read
#   - encoding: Encoding of the CSV file, set to "ISO-8859-1"
data = pd.read_csv(r"sms_raw_NB.csv", encoding="ISO-8859-1")

# Mapping the 'type' column to numeric values 1 and 0
# The 'spam' column is created where the value is 1 for 'spam' messages and 0 
# for other messages
# The np.where() function is used to conditionally assign values based on a condition
data['spam'] = np.where(data.type == 'spam', 1, 0)

###############################################################################
# Importing the create_engine function from the sqlalchemy module
from sqlalchemy import create_engine, text
from urllib.parse import quote


# MySQL
# Credentials to connect to Database
user = 'user1'  # user name
pw = quote('amer@mysql')  # password
db = 'titan'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}") # Creating a database engine to connect


data.to_sql('sms_raw', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

###############################

###############################
# PostgreSQL
# pip install psycopg2 

# Creating an engine that connects to PostgreSQL
# conn_string = psycopg2.connect(database = "postgres", user = 'postgres', password = 'monish1234', host = 'localhost', port= '5432')

# Defining the connection string for connecting to the PostgreSQL database
# Parameters:
#   - user: PostgreSQL username
#   - pw: PostgreSQL password
#   - localhost: PostgreSQL server address
#   - db: PostgreSQL database name
conn_string = ("postgresql+psycopg2://{user}:{pw}@localhost/{db}"
               .format(user="postgres",  # PostgreSQL username
                       pw=quote("amer@ps"),  # PostgreSQL password
                       db="postgres"))      # PostgreSQL database name

# Creating a database engine using the connection string
engine = create_engine(conn_string)



# Writing the DataFrame 'data' to a table named 'sms_raw' in the PostgreSQL database
# Parameters:
#   - 'sms_raw': Name of the table in the database
#   - con: Connection object to the database
#   - if_exists: Behavior when the table already exists ('replace' will replace it)
#   - index: Whether to write the DataFrame index as a column (set to False)
data.to_sql('sms_raw', con=engine.connect(), if_exists='replace', index=False)

###############################################################################


# Data Ingestion from DB
# Select query - Defining the SQL query to retreive data from the 'sms_raw' table
sql = 'SELECT * FROM sms_raw'


# Executing the SQL query and reading the result into a DataFrame 'email_data'
# # Establishing a connection to the database and reading the table
email_data = pd.read_sql_query(text(sql), engine.connect())


# From PostgreSQL
# email_data = pd.read_sql_query(sql, conn)

 
# Data Preprocessing - textual data

# Checking for class imbalance by counting the occurrences of each value in the 'type' column
# and calculating the percentage of each value
print(email_data['type'].value_counts())
print(email_data['type'].value_counts() / len(email_data['type']))

# Alternatively, using groupby to achieve the same result
print(email_data.groupby(['type']).size())
print(email_data.groupby(['type']).size() / len(email_data['type']))

# Splitting the data into training and testing sets
# Parameters:
#   - email_data: DataFrame containing the data to be split
#   - test_size: Proportion of the dataset to include in the test split (set to 20%)
#   - stratify: Ensures that the class distribution is preserved in the train/test splits
#   - random_state: Seed for random number generation for reproducibility
email_train, email_test = train_test_split(email_data, test_size=0.2, stratify=email_data[['spam']], random_state=0)

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

countvectorizer = CountVectorizer(analyzer = 'word', stop_words = 'english')

###########################
# for illustrative purposes
s_sample = email_train.loc[email_train.text.str.len() < 50].sample(3, random_state = 35)
s_sample = s_sample.iloc[:, 0:2]

# Document Term Matrix with CountVectorizer (# returns 1D array)
s_vec = pd.DataFrame(countvectorizer.fit_transform(s_sample.values.ravel()).\
        toarray(), columns = countvectorizer.get_feature_names_out())

s_vec
###########################    

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email texts into word count matrix format - Bag of Words
emails_bow = CountVectorizer(analyzer = split_into_words).fit(email_data.text)

# Defining BOW for all messages
all_emails_matrix = emails_bow.transform(email_data.text)

# For training messages
train_emails_matrix = emails_bow.transform(email_train.text)

# For testing messages
test_emails_matrix = emails_bow.transform(email_test.text)



# We will use the SMOTE technique to handle class imbalance.
# Oversampling can be a good option when we have a class imbalance.
# Due to this our model will perform poorly in capturing variation in a class
# because we have too few instances of that class, relative to one or more other classes.

# SMOTE: Is an approach to oversample (duplicating examples) the minority class
# This is a type of data augmentation for the minority class and is referred 
# to as the Synthetic Minority Oversampling Technique, or SMOTE for short.
smote = SMOTE(random_state = 0)

# Transform the dataset
# Resampling the training data using SMOTE to address class imbalance
# Parameters:
#   - train_emails_matrix: Feature matrix of the training data
#   - email_train.spam: Target variable (spam or not spam) of the training data
X_train, y_train = smote.fit_resample(train_emails_matrix, email_train.spam)

# Checking unique values in the resampled target variable
print(y_train.unique())

# Counting the number of '1's (spam)
print(y_train.values.sum())

# Counting the number of '0's (not spam)
print(y_train.size - y_train.values.sum())
# The data is now balanced

# Instantiating the Multinomial Naive Bayes classifier with default parameters
classifier_mb = MultinomialNB()

# Training the classifier on the resampled training data
classifier_mb.fit(X_train, y_train)

# Making predictions on the test data
test_pred_m = classifier_mb.predict(test_emails_matrix)

# Generating a confusion matrix to evaluate predictions on the test data
confusion_matrix_test_m = pd.crosstab(email_test.spam, test_pred_m)
confusion_matrix_test_m

# Calculating accuracy on the test data
accuracy_test_m = np.mean(test_pred_m == email_test.spam)
accuracy_test_m

# Alternatively, calculating accuracy using sklearn's accuracy_score function
accuracy_test_m_alt = skmet.accuracy_score(email_test.spam, test_pred_m)

# Generating a confusion matrix to evaluate predictions on the training data
train_pred_m = classifier_mb.predict(train_emails_matrix)
confusion_matrix_train_m = pd.crosstab(email_train.spam, train_pred_m)

# Calculating accuracy on the training data
accuracy_train_m = np.mean(train_pred_m == email_train.spam)
accuracy_train_m

# Alternatively, calculating accuracy using sklearn's accuracy_score function
accuracy_train_m_alt = skmet.accuracy_score(email_train.spam, train_pred_m)


############################################
# Model Tuning - Hyperparameter optimization

# Multinomial Naive Bayes changing default alpha for Laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# The smoothing process mainly solves the emergence of a zero probability problem in the dataset.

# formula: 
# P(w|spam) = (num of spam with w + alpha)/(Total num of spam emails + K(alpha))
# K = total number of words in the email to be classified

# Saving the Best Model using Pipelines
# Building a pipeline for text classification
# Instantiating a Multinomial Naive Bayes classifier with Laplace smoothing (alpha = 5)
nb = MultinomialNB()

# Defining a pipeline including CountVectorizer, SMOTE for oversampling, and Naive Bayes classifier
pipe1 = make_pipeline(countvectorizer, smote, nb)

# Define parameter grid for GridSearchCV
param_grid = {
    'multinomialnb__alpha': [0.1, 0.5, 1.0, 5.0]  # Adjust alpha values as needed
}
 

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=pipe1, param_grid=param_grid, cv=5)
grid_search.fit(email_train.text, email_train.spam)

# Get best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Use the best estimator to make predictions
best_estimator = grid_search.best_estimator_

# Saving the trained model
joblib.dump(best_estimator, 'processed1')

# Loading the saved model for predictions
model = joblib.load('processed1')

test_pred_best = model.predict(email_test.text)
# Evaluate the predictions
accuracy_test_best = skmet.accuracy_score(email_test.spam, test_pred_best)
print("Accuracy with Best Estimator:", accuracy_test_best)

# Visualize confusion matrix
cm_best = skmet.confusion_matrix(email_test.spam, test_pred_best)
cmplot_best = skmet.ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=['Not Spam', 'Spam'])
cmplot_best.plot()
cmplot_best.ax_.set(title='Spam Detection Confusion Matrix (Best Estimator)',
               xlabel='Predicted Value', ylabel='Actual Value')
















 
