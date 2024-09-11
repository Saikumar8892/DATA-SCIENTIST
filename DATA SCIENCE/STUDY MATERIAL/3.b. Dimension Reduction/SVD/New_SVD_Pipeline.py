'''Dimension Reduction - SVD 
Business Problem: Processing huge files (e.g., images) for real time applications is not feasible. 

# CRISP-ML(Q):
    Business & Data Understanding:
        Business Problem: Huge files to be analyzed requires a lot of compute and is time consuming
        Business Objective: Minimize the compute & time for processing
        Business Constraints: Minimize the low resolution images
        
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

# Install the required packages if not available

# !pip install feature_engine
# !pip install dtale

# **Importing required packages**
# Importing necessary libraries
import numpy as np  # Importing NumPy library for numerical operations
import pandas as pd  # Importing Pandas library for data manipulation and analysis
import matplotlib.pyplot as plt  # Importing Matplotlib library for plotting graphs and visualizations
from sklearn.impute import SimpleImputer  # Importing SimpleImputer from Scikit-learn for data imputation
from sklearn.preprocessing import StandardScaler  # Importing StandardScaler from Scikit-learn for feature scaling
from sklearn.pipeline import make_pipeline  # Importing make_pipeline from Scikit-learn for creating pipeline of preprocessing steps
from sklearn.decomposition import TruncatedSVD  # Importing TruncatedSVD from Scikit-learn for Truncated Singular Value Decomposition
from kneed import KneeLocator  # Importing KneeLocator for finding the knee/elbow point in a curve
from sqlalchemy import create_engine  # Importing create_engine for creating a connection to the database
from urllib.parse import quote # Importing quote to read the urls/passwords having a special charectors

# Database credentials
user = 'root'  # Database username
pw = quote('Sai@123kumar')  # Database password
db = 'pca_svd'  # Database name

# Creating engine to connect to the database using SQLAlchemy
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# **Import the data**
# Reading the dataset "University_Clustering.xlsx" and storing it in the DataFrame 'University'
# Note: Adjust the file path accordingly
University = pd.read_excel(r"University_Clustering.xlsx")

# Displaying the contents of the University dataframe
University

# Dumping the data into a database table named 'university_clustering' using SQLAlchemy
# The 'if_exists' parameter is set to 'replace' to replace the table if it already exists
# 'chunksize' parameter specifies the number of rows to be written at a time
# 'index' parameter is set to False to prevent writing the dataframe index as a column in the table
University.to_sql('university_clustering', con=engine, if_exists='replace', chunksize=1000, index=False)

# SQL query to select all records from the 'university_clustering' table
sql = 'select * from university_clustering'

# Executing the SQL query and loading the results into a Pandas dataframe 'df'
df = pd.read_sql_query(sql, con=engine)

# Printing the dataframe 'df' containing the data loaded from the database
print(df)

# EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS

# Descriptive Statistics and Data Distribution Function
# Generating descriptive statistics of the dataframe 'df1' using the describe() method
# This provides summary statistics of the numerical columns in the dataframe
df.describe()

# Data Preprocessing

# Dropping the unwanted feature 'UnivID' from the dataframe 'df' and storing the result in 'df1'
# Axis=1 specifies that we are dropping columns, not rows
df1 = df.drop(["UnivID"], axis=1)

# Displaying concise summary of the dataframe 'df1' after dropping the 'UnivID' column
df1.info()

# Checking for null values in the dataframe 'df1' and summing them up for each column
# The isnull() method returns a boolean dataframe indicating where values are null
# The sum() method then calculates the sum of True values for each column
df1.isnull().sum()

# Selecting numeric features from the dataframe 'df1' by excluding columns with data type 'object'
numeric_features = df1.select_dtypes(exclude=['object']).columns

# Displaying the names of the selected numeric features
numeric_features

# Make Pipeline

# Define the Pipeline steps

# Define TruncatedSVD model with 5 components
svd = TruncatedSVD(n_components=5)

# Creating a pipeline for numerical data preprocessing:
# 1. Imputation of missing values using mean strategy
# 2. Standardization of data to address scale differences
# 3. Truncated Singular Value Decomposition (SVD) for dimensionality reduction
num_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), svd)

# Displaying the pipeline object
num_pipeline

# Passing the raw data through the pipeline for preprocessing
# The fit method applies each step of the pipeline sequentially to the data
# It returns the transformed data after applying all the steps
processed = num_pipeline.fit(df1[numeric_features]) 

# Displaying the processed data (though not typically stored in a variable)
processed

# Saving the end-to-end SVD pipeline (including imputation and standardization) using joblib
# The processed object contains the pipeline after fitting it to the data
# 'svd_DimRed' is the name of the file to which the pipeline will be saved
import joblib
joblib.dump(processed, 'svd_DimRed')

# Getting the current working directory
# This will display the path where the file 'svd_DimRed' has been saved
import os 
os.getcwd()

# Importing the saved pipeline using joblib
# Loading the pipeline from the file 'svd_DimRed'
model = joblib.load("svd_DimRed")

# Displaying the imported pipeline
model

# Applying the saved SVD pipeline on the original dataset to extract SVD values
# The transform method applies the preprocessing steps defined in the pipeline to the input data
# The result is stored in a Pandas DataFrame called svd_res
svd_res = pd.DataFrame(model.transform(df1[numeric_features]))

# Displaying the DataFrame containing the SVD values extracted from the dataset
svd_res

# Extracting SVD weights (components) from the saved model
# The components_ attribute of the SVD model contains the principal axes in feature space
# Each row represents a principal component, and each column represents a feature
svd.components_

# Printing the variance percentage explained by each component
# The explained_variance_ratio_ attribute of the SVD model contains the ratio of variance explained by each component
print(svd.explained_variance_ratio_)

# Calculating the cumulative explained variance percentage
# np.cumsum() computes the cumulative sum of the explained variance ratios
var1 = np.cumsum(svd.explained_variance_ratio_)

# Printing the cumulative explained variance percentage
print(var1)

# Plotting the variance explained by each SVD component as a function of the number of components
# This plot helps visualize how much variance in the data is explained by each additional SVD component
# The x-axis represents the number of components, and the y-axis represents the cumulative explained variance percentage
plt.plot(var1, color="red")

# Adding comments to the plot
plt.xlabel('Number of Components')  # Labeling the x-axis
plt.ylabel('Cumulative Explained Variance (%)')  # Labeling the y-axis
plt.title('Variance Plot for SVD Components')  # Adding title to the plot

# Displaying the plot
plt.show()

# KneeLocator
# Importing the KneeLocator class from the kneed module
from kneed import KneeLocator

# Creating a KneeLocator object to identify the knee/elbow point in the cumulative explained variance plot
# range(len(var1)) represents the x-values (number of components)
# var1 represents the y-values (cumulative explained variance percentage)
# curve='concave' specifies the type of curve to fit (concave or convex)
# direction='increasing' indicates the direction of the curve
kl = KneeLocator(range(len(var1)), var1, curve='concave', direction="increasing") 

# Accessing the identified knee/elbow point from the KneeLocator object
kl.elbow

# Setting the style of the plot to seaborn for better aesthetics
plt.style.use("ggplot")

# Plotting the cumulative explained variance percentage as a function of the number of components
plt.plot(range(len(var1)), var1)

# Setting x-ticks to be at integer positions (number of components)
plt.xticks(range(len(var1)))

# Labeling the y-axis as "variance"
plt.ylabel("Cumulative Explained Variance (%)")

# Adding a vertical line at the knee/elbow point identified by KneeLocator
plt.axvline(x=kl.elbow, color='r', label='axvline - full height', ls='--')

# Displaying the plot
plt.show()

# SVD for Feature Extraction
# Creating the final dataset with a manageable number of columns (Feature Extraction)

# Concatenating the 'Univ' column from the original dataframe 'df' 
# with the first three SVD components from the SVD results 'svd_res'
# along the columns (axis=1) to create the final dataframe
final = pd.concat([df.Univ, svd_res.iloc[:, 0:3]], axis=1)

# Renaming the columns of the final dataframe for clarity
final.columns = ['Univ', 'svd0', 'svd1', 'svd2']

# Displaying the final dataframe containing the university names and SVD components
final

# Scatter diagram

# Plotting a scatter diagram of svd0 vs svd1 using the final dataframe 'final'
# figsize specifies the size of the plot
ax = final.plot(x='svd0', y='svd1', kind='scatter', figsize=(12, 8))

# Adding annotations (university names) to the scatter plot
# lambda function is used to apply the text annotation to each point in the plot
final[['svd0', 'svd1', 'Univ']].apply(lambda x: ax.text(*x), axis=1)

# Prediction on new data

# Reading the new data from the Excel file "new_Univ_4_pred.xlsx" and storing it in the DataFrame 'newdf'
newdf = pd.read_excel(r"new_Univ_4_pred.xlsx")

# Displaying the contents of the new DataFrame 'newdf'
newdf

# Dropping the unwanted feature 'UnivID' from the new dataframe and storing the result in 'newdf1'
newdf1 = newdf.drop(["UnivID"], axis=1)

# Selecting numeric features from the new dataframe 'newdf1' by excluding columns with data type 'object'
num_feat = newdf1.select_dtypes(exclude=['object']).columns

# Displaying the names of the selected numeric features
num_feat

# Transforming the new data using the previously saved SVD model
# Transforming the numeric features of 'newdf1' using the model and storing the result in 'new_res'
new_res = pd.DataFrame(model.transform(newdf1[num_feat]))

# Displaying the transformed new data
new_res
