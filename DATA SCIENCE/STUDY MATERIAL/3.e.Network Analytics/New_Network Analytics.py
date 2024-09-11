# # Problem Statement: -
# There are two datasetsconsisting of information for the connecting routes and flight halt. Create network analytics models on both the datasets separately and measure degree centrality, degree of closeness centrality, and degree of in-between centrality.
# ●	Create a network using edge list matrix(directed only).
# ●	Columns to be used in R:
# 
# Flight_halt=c("ID","Name","City","Country","IATA_FAA","ICAO","Latitude","Longitude","Altitude","Time","DST","Tz database time")
# 
# connecting routes=c("flights", " ID", "main Airport”, “main Airport ID", "Destination ","Destination  ID","haults","machinary")
# 

# # network analytics:-
# Network analytics is the application of big data principles and tools to the management and security of data networks.
# 

import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from urllib.parse import quote

# Creating engine which link to SQL via python
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",# user
                               pw=quote("Sai@123kumar"), # passwrd
                               db="network_analysis")) #database

# Reading data from loacal drive
connecting_routes = pd.read_csv(r"connecting_routes.csv")
connecting_routes.head()

# Loading data into sql database
connecting_routes.to_sql('connecting_routes', con = engine, if_exists = 'replace', chunksize = 1000, index= False)

# Define the SQL query to select all records from the 'connecting_routes' table
sql = 'select * from connecting_routes;'

# Execute the SQL query and read the result into a pandas DataFrame using the SQLAlchemy engine
connecting_routes = pd.read_sql_query(sql, con=engine)

# Display the first few rows of the DataFrame
connecting_routes.head()

# Select a subset of the DataFrame (rows 0 to 149, and columns 1 to 7)
connecting_routes = connecting_routes.iloc[0:150, 1:8]

# Get the column names of the subset DataFrame
connecting_routes.columns

# Create an empty graph object using NetworkX
for_g = nx.Graph()

# Populate the graph with edges from the DataFrame, specifying the source and target columns
for_g = nx.from_pandas_edgelist(connecting_routes, source='source airport', target='destination apirport')

# Print information about the graph, such as number of nodes and edges
print(for_g)
print('Number of nodes', len(for_g.nodes))
print('Number of edges', len(for_g.edges))
print('Average degree', sum(dict(for_g.degree).values()) / len(for_g.nodes))
# #  centrality:-
# 
# 
# **Degree centrality** is defined as the number of links incident upon a node (i.e., the number of ties that a node has). ... Indegree is a count of the number of ties directed to the node (head endpoints) and outdegree is the number of ties that the node directs to others (tail endpoints).
# 
# **Eigenvector Centrality** The adjacency matrix allows the connectivity of a node to be expressed in matrix form. So, for non-directed networks, the matrix is symmetric.Eigenvector centrality uses this matrix to compute its largest, most unique eigenvalues.
# 
# **Closeness Centrality** An interpretation of this metric, Centralness.
# 
# **Betweenness centrality** This metric revolves around the idea of counting the number of times a node acts as a bridge.


# Create a DataFrame to store centrality measures
data = pd.DataFrame({
    "closeness": pd.Series(nx.closeness_centrality(for_g)),  # Calculate closeness centrality for each node
    "Degree": pd.Series(nx.degree_centrality(for_g)),  # Calculate degree centrality for each node
    "eigenvector": pd.Series(nx.eigenvector_centrality(for_g, max_iter=600)),  # Calculate eigenvector centrality for each node
    "betweenness": pd.Series(nx.betweenness_centrality(for_g))  # Calculate betweenness centrality for each node
}) 

# Display the DataFrame containing centrality measures
data

# Create an empty graph object using NetworkX
for_g = nx.Graph()

# Populate the graph with edges from the DataFrame, specifying the source and target columns
for_g = nx.from_pandas_edgelist(connecting_routes, source='source airport', target='destination apirport')

# Create a new figure for the plot
f = plt.figure()

# Define the layout of the nodes using the spring layout algorithm with a specified spring constant (k)
pos = nx.spring_layout(for_g, k=0.015)

# Draw the network graph using NetworkX, specifying the graph, layout, axis, node size, and node color
nx.draw_networkx(for_g, pos, ax=f.add_subplot(111), node_size=15, node_color='red')

# Display the plot
plt.show()

# Save the figure as a PNG file (optional)
# f.savefig("graph.png")



