# DBScan demonstration on Spherical Data. 
# Compare the applications of Agglomerative, Kmeans, and DBScan clustering techniques

# Importing necessary libraries
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN  # Importing clustering algorithms
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting
import pandas as pd  # Importing Pandas for data manipulation

# Reading the CSV file 'Dbscan_spherical_data.csv' into a DataFrame 'df'
df = pd.read_csv(r"Dbscan_spherical_data.csv")

# Displaying the DataFrame 'df' to inspect its contents
df

# Creating a scatter plot of the data points using the 'x' and 'y' columns from DataFrame 'df'
plt.scatter(df.x, df.y)

# Displaying the scatter plot
plt.show()

# Creating an instance of AgglomerativeClustering with 5 clusters and using 'average' linkage method
ac = AgglomerativeClustering(5, linkage='average')

# Fitting the AgglomerativeClustering model to the data and predicting the cluster labels for each sample
ac_clusters = ac.fit_predict(df)

# Plotting the clusters obtained from Agglomerative Clustering
plt.figure(1)
plt.title("Clusters from Agglomerative Clustering")
plt.scatter(df.x, df.y, c=ac_clusters, s=50, cmap='tab20b')
plt.show()


# Creating an instance of KMeans clustering with 5 clusters
km = KMeans(5)

# Fitting the KMeans model to the data and predicting the cluster labels for each sample
km_clusters = km.fit_predict(df)

# Plotting the clusters obtained from K-Means
plt.figure(2)
plt.title("Clusters from K-Means")
plt.scatter(df.x, df.y, c=km_clusters, s=50, cmap='tab20b')
plt.show()


# Creating an instance of DBSCAN with epsilon (eps) value of 0.2 and minimum number of samples (min_samples) set to 11
db = DBSCAN(eps=0.2, min_samples=11).fit(df)

# Extracting the cluster labels assigned by DBSCAN
labels = db.labels_

# Plotting the clusters obtained from DBSCAN
plt.figure(3)
plt.scatter(df.x, df.y, c=labels, cmap='tab20b')
plt.title("DBSCAN from Scratch Performance")
plt.show()
