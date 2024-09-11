'''
# K-Means Clustering Algorithm - Data Mining (Machine Learning) Unsupervised learning Algorithm

# Business Problem Statement:

# Students have to evaluate a lot of factors before taking a decision 
to join a university for their higher education requirements.

# `CRISP-ML(Q)` process model describes six phases:
# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Evaluation
# 5. Deployment
# 6. Monitoring and Maintenance
'''
# Objective(s): Maximize the convenience of admission process
# Constraints: Minimize the brain drain

'''Success Criteria'''

# Business Success Criteria: Reduce the application process time from anywhere between 20% to 40%
# ML Success Criteria: Achieve Silhoutte coefficient of atleast 0.5
# Economic Success Criteria: US Higher education department will see an increase in revenues by atleast 30%

# **Proposed Plan:**
# Grouping the available universities will allow to understand the characteristics of each group.

'''
# ## Data Collection

# Data: 
#    The university details are obtained from the US Higher Education Body and is publicly available for students to access.
# 
# Data Dictionary:
# - Dataset contains 25 university details
# - 8 features are recorded for each university
# 
# Description:
# - Univ - University Name
# - State - Location (state) of the university
# - SAT - Cutoff SAT score for eligibility
# - Top10 - % of students who ranked in top 10 in their previous academics
# - Accept - % of students admitted to the universities
# - SFRatio - Student to Faculty ratio
# - Expenses - Overall cost in USD
# - GradRate - % of students who graduate
'''

# #### Install the required packages if not available
# !pip install feature_engine
# !pip install sklearn_pandas

# **Importing required packages**
# import numpy as np
import pandas as pd  # Importing pandas library for data manipulation and analysis
import sweetviz  # Importing sweetviz for automated exploratory data analysis
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

from sklearn.pipeline import Pipeline  # Importing Pipeline for chaining preprocessing steps
from sklearn.impute import SimpleImputer  # Importing SimpleImputer for handling missing values
from sklearn.preprocessing import MinMaxScaler  # Importing MinMaxScaler for feature scaling
from sklearn.preprocessing import OrdinalEncoder # Importing OrdinalEncoder for converting string to integer
from sklearn.compose import ColumnTransformer # Importing ColumnTransformer to transfer pipelines into the data 

from sklearn.cluster import KMeans  # Importing KMeans for clustering
from sklearn import metrics  # Importing metrics for evaluating clustering performance
import joblib  # Importing joblib for saving trained models
import pickle  # Importing pickle for saving Python objects
from sqlalchemy import create_engine, text
from urllib.parse import quote
# Importing the data from an Excel file
uni = pd.read_csv(r"Sample - Superstore.csv",encoding='windows-1254')

# Credentials to connect to the database
user = 'root'  # Username
pw = quote('Sai@123kumar')  # Password
db = 'cluster_demo'  # Database name

# Creating a database engine to connect to the MySQL database using the provided credentials
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Using the to_sql() function to push the DataFrame 'uni' onto a SQL table named 'univ_tbl' in the database
# The 'if_exists' parameter is set to 'replace' to replace the table if it already exists
# The 'chunksize' parameter specifies the number of rows to write at a time
# The 'index' parameter is set to False to avoid writing row indices to the SQL table
uni.to_sql('univ_tbl', con=engine, if_exists='replace', chunksize=1000, index=False)

# Defining a SQL query to select all records from the 'univ_tbl' table
sql = text('select * from univ_tbl;')

# Executing the SQL query and reading the results into a DataFrame 'df' using read_sql_query() function
df = pd.read_sql_query(sql, engine.connect())

# Displaying the data types and non-null counts of each column in the DataFrame 'df'
df.info()

# Dropping the unwanted features "UnivID" and "Univ" from the DataFrame 'df' and creating a new DataFrame 'df1'
df1 = df.drop(["UnivID", "Univ"], axis=1)

# Displaying the first few rows of the DataFrame 'df1' after dropping the unwanted features
df1.head()


# # EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS

# ***Descriptive Statistics and Data Distribution Function***
# Generating descriptive statistics of the DataFrame 'df1', including count, mean, standard deviation, minimum, maximum, and quartile values for numerical columns
df1.describe()

# Checking unique values for the categorical feature 'State' in the DataFrame 'df1'
# The unique() method returns an array of unique values
unique_states = df1.State.unique()

# Counting the number of unique states
num_unique_states = df1.State.unique().size

# Counting the occurrences of each unique state and displaying them in descending order
state_value_counts = df1.State.value_counts()

# AutoEDA
# Automated Libraries
# import sweetviz
# Generating a report using Sweetviz to analyze the DataFrame 'df1' and comparing it with itself (duplicate)
my_report = sweetviz.analyze([df1, "df1"])

# Displaying the Sweetviz report as an HTML file named 'Report.html'
my_report.show_html('Report.html')

# Checking for missing data in the DataFrame 'df1'
# The isnull() method returns a DataFrame of boolean values indicating whether each element is missing
# The sum() method sums up the missing values for each column
# This provides the count of missing values in each column
missing_data = df1.isnull().sum()

# Segregate Numeric and Non-numeric columns
df1.info()

# Selecting numeric columns (excluding object dtype) from the DataFrame 'df1' and storing their column names in 'numeric_features'
numeric_features = df1.select_dtypes(exclude=['object']).columns

# Displaying the numeric features
numeric_features

# Selecting non-numeric columns (object dtype) from the DataFrame 'df1' and storing their column names in 'categorical_features'
categorical_features = df1.select_dtypes(include=['object']).columns

# Displaying the non-numeric features
categorical_features

# Defining a Pipeline to deal with missing data and scaling numeric columns
# The Pipeline consists of two steps: imputation using mean strategy and scaling using MinMaxScaler
num_pipeline = Pipeline([('impute', SimpleImputer(strategy='mean')), ('scale', MinMaxScaler())])

# Displaying the defined pipeline
num_pipeline

# Encoding Non-numeric fields
# Defining a pipeline to convert categorical data into numeric data 
categ_pipeline = Pipeline([('OnehotEncode', OrdinalEncoder())])

# Displaying the defined pipeline
categ_pipeline

# Using ColumnTransfer to transform the Pipelines into the data. 
# This estimator allows different columns or column subsets of the input to be
# transformed separately and the features generated by each transformer will
# be concatenated to form a single feature space.
preprocess_pipeline = ColumnTransformer([('categorical', categ_pipeline, categorical_features), 
                                       ('numerical', num_pipeline, numeric_features)], 
                                        remainder = 'passthrough') # Skips the transformations for remaining columns

preprocess_pipeline

# Pass the raw data through pipeline
processed = preprocess_pipeline.fit(df1) 


# ## Save the Imputation and Encoding pipeline
# import joblib
joblib.dump(processed, 'preprocessing')

# File gets saved under current working directory
import os
os.getcwd()

# Clean and processed data for Clustering
univ_clean = pd.DataFrame(processed.transform(df1), columns = processed.get_feature_names_out())
univ_clean

# Clean data
univ_clean.describe()

# # CLUSTERING MODEL BUILDING

# ### KMeans Clustering
# Libraries for creating scree plot or elbow curve 
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

###### scree plot or elbow curve ############
TWSS = []  # List to store the total within-cluster sum of squares (TWSS) for each value of k
k = list(range(2, 9))  # List of values of k (number of clusters) to be evaluated

# Looping through each value of k
for i in k:
    kmeans = KMeans(n_clusters=i)  # Creating a KMeans clustering model with i clusters
    kmeans.fit(univ_clean)  # Fitting the KMeans model to the cleaned numeric data
    TWSS.append(kmeans.inertia_)  # Appending the total within-cluster sum of squares (TWSS) to the list TWSS

# Displaying the TWSS values for each value of k
TWSS

# Creating a scree plot to visualize the relationship between the number of clusters and TWSS
plt.plot(k, TWSS, 'ro-')  # Plotting the values of k (x-axis) against the TWSS (y-axis)
plt.xlabel("No_of_Clusters")  # Labeling the x-axis as "No_of_Clusters"
plt.ylabel("total_within_SS")  # Labeling the y-axis as "total_within_SS"

# see the styles available in matplotlib
print(plt.style.available)

# ## Using KneeLocator
List = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters = k, init = "random", max_iter = 30, n_init = 10) 
    kmeans.fit(univ_clean)
    List.append(kmeans.inertia_)

# !pip install kneed
from kneed import KneeLocator
# kl = KneeLocator(range(2, 9), List, curve = 'convex')
kl = KneeLocator(range(2, 9), List, curve='convex', direction = 'decreasing')
kl.elbow
plt.style.use("ggplot")
plt.plot(range(2, 9), List)
plt.xticks(range(2, 9))
plt.ylabel("Interia")
plt.axvline(x = kl.elbow, color = 'r', label = 'axvline - full height', ls = '--')
plt.show() 

# Not able to detect the best K value (knee/elbow) as the line is mostly linear

# Creating a KMeans clustering model with 3 clusters
model = KMeans(n_clusters=4)

# Fitting the KMeans model to the cleaned numeric data 'univ_clean'
yy = model.fit(univ_clean)

# Obtaining the cluster labels assigned by the KMeans model to each data point
cluster_labels = model.labels_


# ## Cluster Evaluation

# **Silhouette coefficient:**  
# Silhouette coefficient is a Metric, which is used for calculating 
# goodness of clustering technique and the value ranges between (-1 to +1).
# It tells how similar an object is to its own cluster (cohesion) compared to 
# other clusters (separation).
# A score of 1 denotes the best meaning that the data point is very compact 
# within the cluster to which it belongs and far away from the other clusters.
# Values near 0 denote overlapping clusters.

# from sklearn import metrics
# Calculating the silhouette score to evaluate the clustering performance
# The silhouette score measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation)
# Higher silhouette scores indicate better clustering results, with a maximum score of 1 indicating dense, well-separated clusters
# A score close to 0 suggests overlapping clusters, and negative scores indicate that data points might have been assigned to the wrong cluster
silhouette_score = metrics.silhouette_score(univ_clean, model.labels_)

# **Calinski Harabasz:**
# Higher value of CH index means cluster are well separated.
# There is no thumb rule which is acceptable cut-off value.
# Calculating the Calinski-Harabasz score to evaluate the clustering performance
# The Calinski-Harabasz score is a ratio of the sum of between-cluster dispersion to the sum of within-cluster dispersion
# Higher Calinski-Harabasz scores indicate better-defined, more separate clusters
# This score is sensitive to cluster density and shape, with well-separated and compact clusters yielding higher scores
calinski_harabasz_score = metrics.calinski_harabasz_score(univ_clean, model.labels_)

# **Davies-Bouldin Index:**
# Unlike the previous two metrics, this score measures the similarity of clusters. 
# The lower the score the better the separation between your clusters. 
# Vales can range from zero and infinity
# Calculating the Davies-Bouldin score to evaluate the clustering performance
# The Davies-Bouldin score measures the average similarity between each cluster and its most similar cluster, where similarity is defined as the ratio of within-cluster scatter to between-cluster separation
# Lower Davies-Bouldin scores indicate better clustering results, with values closer to 0 indicating well-separated clusters and higher values suggesting more overlapping clusters
# Unlike the silhouette score, the Davies-Bouldin score does not require ground truth labels and is more robust to noise and outliers
davies_bouldin_score = metrics.davies_bouldin_score(univ_clean, model.labels_)

# ### Evaluation of Number of Clusters using Silhouette Coefficient Technique
from sklearn.metrics import silhouette_score  # Importing the silhouette score function from sklearn.metrics

silhouette_coefficients = []  # List to store silhouette coefficients for different values of k

# Looping through each value of k from 2 to 8
for k in range(2, 9):
    # Creating a KMeans clustering model with k clusters
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(univ_clean)  # Fitting the KMeans model to the cleaned numeric data 'univ_clean'
    
    # Calculating the silhouette score for the current value of k
    score = silhouette_score(univ_clean, kmeans.labels_)
    
    # Storing the value of k and its corresponding silhouette coefficient in a list
    silhouette_coefficients.append([k, score])

# Displaying the list of silhouette coefficients for different values of k
silhouette_coefficients

# Sorting the list of silhouette coefficients in descending order based on the silhouette coefficient value
# This allows us to identify the optimal value of k that maximizes the silhouette coefficient
sorted(silhouette_coefficients, reverse=True, key=lambda x: x[1])


# silhouette coefficients shows the number of clusters 'k = 2' as the best value

# Building KMeans clustering
bestmodel = KMeans(n_clusters=2)  # Creating a KMeans clustering model with 2 clusters
result = bestmodel.fit(univ_clean)  # Fitting the KMeans model to the cleaned numeric data 'univ_clean'

# Saving the trained KMeans clustering model to a file named 'Clust_Univ.pkl' using pickle
import pickle
pickle.dump(result, open('Clust_Univ.pkl', 'wb'))

import os
os.getcwd()  # Getting the current working directory

# Obtaining the cluster labels assigned by the KMeans model to each data point
cluster_labels = bestmodel.labels_

mb = pd.Series(cluster_labels)  # Converting the cluster labels to a pandas Series

# Concatenating the cluster labels with the original University names and cleaned numeric data
df_clust = pd.concat([mb, df.Univ, df1], axis=1)  # Concatenating along the columns axis
df_clust = df_clust.rename(columns={0: 'cluster_id'})  # Renaming the cluster label column to 'cluster_id'
df_clust.head()  # Displaying the first few rows of the DataFrame 'df_clust'

# Aggregate using the mean of each cluster
# Grouping the DataFrame 'df_clust' by cluster_id and calculating the mean of numeric columns for each cluster
cluster_agg = df_clust.iloc[:, 3:].groupby(df_clust.cluster_id).mean()

# Displaying the aggregated cluster-level statistics
cluster_agg

# Saving the results DataFrame 'df_clust' to a CSV file named 'KMeans_University.csv'
df_clust.to_csv('KMeans_University.csv', encoding='utf-8', index=False)

# Importing the os module to work with the operating system
import os

# Getting the current working directory
os.getcwd()



'''
# ## Note:
# ### The trained KMeans clustering model works for Universities from the following states only:
PA
CA
NY
MA
IL
IN
RI
TX
WI
VA
MI
MD
NJ
DC
NC
NH
CT
'''