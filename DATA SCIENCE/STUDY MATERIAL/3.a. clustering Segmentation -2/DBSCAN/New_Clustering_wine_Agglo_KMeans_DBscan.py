# Problem Statement

'''
# Identifying the best quality wine is a special skill and very few experts are specialized in accurately detecting the quality.
# The objective of this project is to simplify the process of detecting the quality of wine.

# `CRISP-ML(Q)` process model describes six phases:
# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Evaluation
# 5. Model Deployment
# 6. Monitoring and Maintenance

# **Objective(s):** Minimize Shipment Organization Time
# 
# **Constraints:** Minimize Specialists' Dependency    

# **Success Criteria**
# - **Business Success Criteria**: Reduce the time of wine quality check by anywhere between 20% to 40%
# - **ML Success Criteria**: Achieve Silhouette coefficient of atleast 0.5
# - **Economic Success Criteria**: Wine distillers will see an increase in revenues by atleast 20%

# **Proposed Plan:**
# Grouping the available wines will allow to understand the characteristics of each group.

# ### Data Dictionary

# - OD_read: Amount of dilution in that particular wine type
# - Proline: Amount of Proline in that particular wine type 
# Proline is typically the most abundant amino acid present in grape juice and wine
'''

import pandas as pd  # Importing Pandas library for data manipulation
import sweetviz  # Importing Sweetviz library for automated EDA (Exploratory Data Analysis)
import matplotlib.pyplot as plt  # Importing Matplotlib library for plotting

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN  # Importing machine learning algorithms for clustering
from sklearn.metrics import silhouette_score  # Importing silhouette_score for cluster evaluation

from sqlalchemy import create_engine, text  # Importing create_engine and text from sqlalchemy for database interaction

# Load Wine data set from a CSV file into a Pandas DataFrame
df = pd.read_csv(r"wine_data.csv")

from urllib.parse import quote
# Credentials to connect to Database
user = 'root'  # user name
pw = quote('Sai@123kumar')  # password
db = 'cluster_demo'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}") # Creating a database engine to connect

# Using to_sql() function to push the DataFrame 'df' onto a SQL table named 'wine_tbl' in the connected database
# Parameters:
# - 'wine_tbl': Name of the SQL table
# - con = engine: Database engine used for the connection
# - if_exists = 'replace': If the table already exists, it will be replaced
# - chunksize = 1000: Data will be inserted in chunks of 1000 rows
# - index = False: Index will not be included as a column in the SQL table
df.to_sql('wine_tbl', con=engine, if_exists='replace', chunksize=1000, index=False)

# Defining SQL query to select all records from the 'wine_tbl' table
sql = text('select * from wine_tbl;')

# Executing the SQL query using engine.connect() and storing the result in the DataFrame 'wine_df'
wine_df = pd.read_sql_query(sql, engine.connect())

# Displaying the first few rows of the DataFrame 'wine_df'
wine_df.head()

# ## EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS

# ***Descriptive Statistics and Data Distribution Function***
wine_df.describe()  # Generating descriptive statistics of the DataFrame 'wine_df' to analyze the numerical columns

# ***1st Moment Business Decision (Measures of Central Tendency)***
# 1) Mean
# 2) Median
# 3) Mode

# ***2nd Moment Business Decision (Measures of Dispersion)***
# 1) Variance
# 2) Standard deviation
# 3) Range (maximum - minimum)

# ***3rd Business Moment Decision (Skewness)***
# Measure of asymmetry in the data distribution
# wine_df.skew()

# ***4th Business Moment Decision (Kurtosis)***
# Measure of peakedness - represents the overall spread in the data
# wine_df.kurt()


# AutoEDA
# ## Automated Libraries
# import sweetviz
# Generating a comprehensive report using Sweetviz library to analyze the DataFrame 'wine_df'
# The report will be saved as an HTML file named 'Report.html'
my_report = sweetviz.analyze([wine_df, "wine_df"])

# Displaying the HTML report
my_report.show_html('Report.html')


# ## Data Preprocessing and Cleaning

# **Typecasting** :
# 
# As Python automatically interprets the data types, there may be a requirement
# for the data type to be converted. The process of converting one data type
# to another data type is called Typecasting.
# 
# Example: 
# 1) int to float
# 2) float to int
wine_df.info()  # Displaying concise summary of DataFrame 'wine_df', including the number of non-null values and data types of each column


# **Handling duplicates:**
# If the dataset has multiple entries of the same record then we can remove the duplicate entries. In case of duplicates we will use function drop_duplicates()
# Checking for duplicate rows in the DataFrame 'wine_df'
# The duplicated() function returns a Boolean Series where True indicates duplicate rows and False indicates non-duplicate rows.
duplicate = wine_df.duplicated()

print(duplicate)  # Printing the Boolean Series indicating duplicate rows

# Calculating the total number of duplicate rows in the DataFrame 'wine_df'
# The sum() function is used to sum up the True values in the Boolean Series 'duplicate'
sum(duplicate)

print(wine_df.shape)  # Printing the shape (number of rows and columns) of the DataFrame 'wine_df'

# Removing duplicate rows from the DataFrame 'wine_df'
# The drop_duplicates() function returns a DataFrame with duplicate rows removed
wine_df = wine_df.drop_duplicates()

print(wine_df.shape)  # Printing the new shape (number of rows and columns) of the DataFrame 'wine_df' after removing duplicates


# **Missing Value Analysis**

# ***IMPUTATION:***
# The process of dealing with missing values is called Imputation.
# Most popular substitution based Imputation techniques are:
# 1) Mean imputation for numeric data
# 2) Mode imputation for non-numeric data

wine_df.isnull().sum() # Check for missing values


# ### Outliers Analysis:
# Exceptional data values in a variable can be outliers. In case of outliers we can use one of the strategies of 3 R (Rectify, Retain, or Remove)

# **Box Plot**
# Visualize numeric data using boxplot for outliers

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True
wine_df.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()  
# **No outliers observed**

# ## Scatter Plot
# Creating a scatter plot using the values from the first and second columns of the DataFrame 'wine_df'
# The first column (index 0) represents 'OD Reading' and the second column (index 1) represents 'Proline'
plt.scatter(wine_df.values[:, 0], wine_df.values[:, 1])

# Setting the title of the scatter plot
plt.title("Wine Dataset")

# Setting the label for the x-axis
plt.xlabel("OD Reading")

# Setting the label for the y-axis
plt.ylabel("Proline")

# Displaying the scatter plot
plt.show()

# Correlation Coefficient
wine_df.corr()


# Generate clusters using Agglomerative Hierarchical Clustering
# Creating an instance of AgglomerativeClustering with 5 clusters and using 'average' linkage method
ac = AgglomerativeClustering(5, linkage='average')

# Fitting the AgglomerativeClustering model to the data and predicting the cluster labels for each sample
ac_clusters = ac.fit_predict(wine_df)

# Creating an instance of KMeans clustering with 5 clusters
km = KMeans(5)

# Fitting the KMeans model to the data and predicting the cluster labels for each sample
km_clusters = km.fit_predict(wine_df)

# Defining parameter options for DBSCAN clustering
db_param_options = [[20, 5], [25, 5], [30, 5], [25, 7], [35, 7], [40, 5]]

# Iterating through each parameter option and performing DBSCAN clustering
for ep, min_sample in db_param_options:
    db = DBSCAN(eps=ep, min_samples=min_sample)
    db_clusters = db.fit_predict(wine_df)
    
    # Printing the parameter values and silhouette score for each DBSCAN clustering
    print("Eps: ", ep, "Min Samples: ", min_sample)
    print("DBSCAN Clustering: ", silhouette_score(wine_df, db_clusters))

# Performing DBSCAN clustering with the selected parameters (eps=40, min_samples=5)
db = DBSCAN(eps=40, min_samples=5)
db_clusters = db.fit_predict(wine_df)

# Plotting the clusters obtained from Agglomerative Clustering
plt.figure(1)
plt.title("Wine Clusters from Agglomerative Clustering")
plt.scatter(wine_df['OD_read'], wine_df['Proline'], c=ac_clusters, s=50, cmap='tab20b')
plt.show()

# Plotting the clusters obtained from KMeans
plt.figure(2)
plt.title("Wine Clusters from K-Means")
plt.scatter(wine_df['OD_read'], wine_df['Proline'], c=km_clusters, s=50, cmap='tab20b')
plt.show()

# Plotting the clusters obtained from DBSCAN
plt.figure(3)
plt.title("Wine Clusters from DBSCAN")
plt.scatter(wine_df['OD_read'], wine_df['Proline'], c=db_clusters, s=50, cmap='tab20b')
plt.show()


# Calculate Silhouette Scores
# Printing a message indicating the start of the section displaying silhouette scores for the Wine Dataset
print("Silhouette Scores for Wine Dataset:\n")

# Calculating and printing the silhouette score for Agglomerative Clustering
print("Agg Clustering: ", silhouette_score(wine_df, ac_clusters))

# Calculating and printing the silhouette score for K-Means Clustering
print("K-Means Clustering: ", silhouette_score(wine_df, km_clusters))

# Calculating and printing the silhouette score for DBSCAN Clustering
print("DBSCAN Clustering: ", silhouette_score(wine_df, db_clusters))

# Saving the DBSCAN model to a file named 'db.pkl' using pickle
import pickle
pickle.dump(db, open('db.pkl', 'wb'))

# Loading the DBSCAN model from the file 'db.pkl' using pickle
model = pickle.load(open('db.pkl', 'rb'))

# Using the loaded DBSCAN model to predict clusters on the Wine Dataset and storing the result
res = model.fit_predict(wine_df)
