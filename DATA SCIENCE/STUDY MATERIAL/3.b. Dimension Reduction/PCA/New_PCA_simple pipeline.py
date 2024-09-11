'''# Dimension Reduction - PCA - In supervised learning (predictive modeling), within regression we have collinearity problem. 
# This can be addressed using PCA. PCs which are the end result of PCA application would be uncorrelated.

# CRISP-ML(Q):
    Business & Data Understanding:
        Business Problem: Huge number of features to analyze requires a lot of compute and is time consuming
        Business Objective: Minimize the compute & time for processing
        Business Constraints: Minimize the features deletion
        
        Success Criteria:
            Business: Reduce the compute required by 50%
            ML: Get at least 50% compression
            Economic: ROI of at least $500K over a period of 1 year

# Data Collection

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
# - SAT - Average SAT score for eligibility
# - Top10 - % of students who ranked in top 10 in their previous academics
# - Accept - % of students admitted to the universities
# - SFRatio - Student to Faculty ratio
# - Expenses - Overall cost in USD
# - GradRate - % of students who graduate'''

# #### Install the required packages if not available

# !pip install feature_engine
# !pip install dtale


# **Importing required packages**
!pip install --upgrade kneed

import numpy as np  # Importing NumPy library for numerical computations
import pandas as pd  # Importing Pandas library for data manipulation
import sweetviz  # Importing Sweetviz library for automated EDA
import matplotlib.pyplot as plt  # Importing Matplotlib library for data visualization
from sklearn.impute import SimpleImputer  # Importing SimpleImputer for handling missing values
from sklearn.preprocessing import StandardScaler  # Importing StandardScaler for feature scaling
from sklearn.pipeline import make_pipeline  # Importing make_pipeline for creating a pipeline of preprocessing steps
from sklearn.decomposition import PCA  # Importing PCA for dimensionality reduction
import joblib # Importing joblib to save pipelines or models
from kneed import KneeLocator  # Importing KneeLocator for finding the knee/elbow point in a curve
from sqlalchemy import create_engine  # Importing create_engine for creating a connection to the database
from urllib.parse import quote # Importing quote to read the urls/passwords having a special charectors

# Database credentials
user = 'root'  # Database username
pw = quote('Sai@123kumar')  # Database password
db = 'pca_svd'  # Database name

# Creating an engine to connect to the database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# **Import the data**
# Reading University data from an Excel file into a DataFrame
University = pd.read_excel(r"University_Clustering.xlsx")

# Displaying the University DataFrame
University

# Dumping University data into a database table named 'university_clustering'
# The table name should be in lower case
University.to_sql('university_clustering', con=engine, if_exists='replace', chunksize=1000, index=False)

# Loading data from the 'university_clustering' table in the database into a DataFrame
sql = 'select * from university_clustering'
df = pd.read_sql_query(sql, con=engine)

# Displaying the DataFrame loaded from the database
print(df)

# Displaying information about the DataFrame, such as column data types and missing values
df.info()

# # EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS
# ***Descriptive Statistics and Data Distribution Function***
# Generating descriptive statistics for the DataFrame and storing the result
res = df.describe()

# Dropping the 'UnivID' column as it is unwanted
df1 = df.drop(["UnivID"], axis=1)

# Displaying information about the DataFrame after dropping 'UnivID' column
df1.info()

# Performing automated Exploratory Data Analysis (AutoEDA) using Sweetviz
# Creating a report to analyze the DataFrame df1 and saving it as an HTML file
my_report = sweetviz.analyze([df1, "df1"])
my_report.show_html('Report.html')

# Checking for missing values in the DataFrame
df1.isnull().sum()

# Displaying information about the DataFrame to check the data types and missing values
df1.info()

# Selecting only numeric features for PCA (Principal Component Analysis)
numeric_features = df1.select_dtypes(exclude=['object']).columns

# Displaying the selected numeric features
numeric_features

# Defining the PCA (Principal Component Analysis) model with 6 components
pca = PCA(n_components=6)

# Make Pipeline

# Creating a pipeline to handle missing values through mean imputation, standardize the data, and perform PCA
# The pipeline consists of SimpleImputer for mean imputation, StandardScaler for data standardization, and PCA for dimensionality reduction
num_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), pca)

# Displaying the pipeline
num_pipeline

# Fitting the raw data to the pipeline, which performs mean imputation, data standardization, and PCA transformation
processed = num_pipeline.fit(df1[numeric_features])

# Displaying the processed data after fitting it to the pipeline
processed

# Applying the pipeline on the original dataset to transform it using mean imputation, standardization, and PCA
univ = pd.DataFrame(processed.transform(df1[numeric_features]))

# Displaying the transformed dataset after applying the pipeline
univ

# Saving the end-to-end PCA pipeline with imputation and standardization using joblib
joblib.dump(processed, 'Data_prep_DimRed')

# Getting the current working directory
import os
os.getcwd()

# Loading the saved pipeline model
model = joblib.load("Data_prep_DimRed")

# Applying the saved model on the dataset to extract PCA values
pca_res = pd.DataFrame(model.transform(df1[numeric_features]))

# Displaying the PCA results
pca_res

# Getting the PCA weights (components) from the saved model
model['pca'].components_

# Storing the PCA weights (components) in a DataFrame for closer inspection
components = pd.DataFrame(model['pca'].components_, columns=numeric_features).T
components.columns = ['pc0', 'pc1', 'pc2', 'pc3', 'pc4', 'pc5']

# Displaying the PCA components
components

# Printing the explained variance ratio of each principal component obtained from PCA
print(model['pca'].explained_variance_ratio_)

# Calculating the cumulative explained variance ratio
var1 = np.cumsum(model['pca'].explained_variance_ratio_)

# Printing the cumulative explained variance ratio
print(var1)

# Plotting the variance explained by PCA components
plt.plot(var1, color="red")

# KneeLocator
# Refer the link to understand the parameters used: https://kneed.readthedocs.io/en/stable/parameters.html     

# from kneed import KneeLocator
# Using the KneeLocator to find the elbow point in the cumulative explained variance plot
kl = KneeLocator(range(len(var1)), var1, curve='concave', direction="increasing")

# Getting the index of the elbow point
kl.elbow

# Setting the style for the plot
plt.style.use("ggplot")

# Plotting the cumulative explained variance ratio
plt.plot(range(len(var1)), var1)

# Setting x-axis ticks
plt.xticks(range(len(var1)))

# Adding labels to axes
plt.ylabel("Variance")
plt.xlabel("Number of Principal Components")

# Adding a vertical line at the elbow point
plt.axvline(x=kl.elbow, color='r', label='Elbow Point', ls='--')

# Displaying the plot
plt.show()

# Kneelocator recommends 3 PCs as the ideal number of features to be considered
# PCA for Feature Extraction

# Final dataset with manageable number of columns (Feature Extraction)

# Concatenating the University names with the first three principal components obtained from PCA
final = pd.concat([df.Univ, pca_res.iloc[:, 0:3]], axis=1)
final.columns = ['Univ', 'pc0', 'pc1', 'pc2']

# Displaying the final DataFrame with University names and principal components
final

# Creating a scatter plot of pc0 vs pc1 with University names as labels
ax = final.plot(x='pc0', y='pc1', kind='scatter', figsize=(12, 8))
final[['pc0', 'pc1', 'Univ']].apply(lambda x: ax.text(*x), axis=1)

# Reading new data for prediction
newdf = pd.read_excel(r"D:/New materials/DS/01-cluestering(k_means)/new_Univ_4_pred.xlsx")

# Dropping unwanted features from the new data
newdf1 = newdf.drop(["UnivID"], axis=1)

# Selecting numeric features for the new data
num_feat = newdf1.select_dtypes(exclude=['object']).columns

# Transforming the new data using the PCA model
new_res = pd.DataFrame(model.transform(newdf1[num_feat]))

# Displaying the transformed new data
new_res

