'''
# Data Mining Unsupervised Learning / Descriptive Modeling - Association Rule Mining

# Problem Statement

# Sales of the bricks and mortar stores has been less in comparison to the 
competitors in the surroundings. 
Store owner realised this by visiting various stores as part of experiments.

# Retail Store (Client) wants to leverage on the transactions data that is being captured. 
The customers purchasing habits needed to be understood by finding the 
association between the products in the customers transactions. 
This information can help Retail Store (client) to determine the shelf placement 
and by devising strategies to increase revenues and develop effective sales strategies.

# `CRISP-ML(Q)` process model describes six phases:
# 
# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Model Evaluation
# 5. Deployment
# 6. Monitoring and Maintenance

# **Objective(s):** Maximize Profits
# 
# **Constraints:** Minimize Marketing Cost

# **Success Criteria**
# 
# - **Business Success Criteria**: Improve the cross selling in Retail Store by 15% - 20%
# 
# - **ML Success Criteria**: Accuracy : NA; 
    Performance : Complete processing within 5 mins on every quarter data
# 
# - **Economic Success Criteria**: Increase the Retail Store profits by atleast 15%
# 
# **Proposed Plan:**
# Identify the Association between the products being purchased by the customers
 from the store

# ## Data Collection

# Data: 
#    The daily transactions made by the customers are captured by the store.
# 
# Description:
# A total of 9835 transactions data captured for the month.
'''

# Mlxtend (machine learning extensions) is a Python library of useful tools for
# the day-to-day data science tasks.

# pip install mlxtend


# Install the required packages if not available
# Install the required packages if not available
import pandas as pd  # Importing pandas library and aliasing it as pd for easier reference.
import matplotlib.pyplot as plt  # Importing matplotlib's pyplot module and aliasing it as plt for easier reference.
from mlxtend.frequent_patterns import apriori, association_rules  # Importing specific functions from mlxtend library.
from mlxtend.preprocessing import TransactionEncoder  # Importing TransactionEncoder from mlxtend.preprocessing.

from sqlalchemy import create_engine  # Importing create_engine function from sqlalchemy library.
from urllib.parse import quote
import pickle  # Importing pickle module for serialization.

# Creating a list named list1 containing a mix of string and integer values.
list1 = ['kumar', '360DigiTMG', 2019, 2022]

# Printing the contents of list1
print(list1)

# Printing the length of list1
print(len(list1))

# Printing the first and last elements of list1
print(list1[0])
print(list1[3])

# Printing the first three elements of list1
print(list1[:3])

# Reversing the order of elements in list1 and printing it
print(list1[::-1])

# Deleting the first element of list1 and printing the updated list
del(list1[0])
print(list1)

# Creating a tuple named tup1
tup1 = ('kumar', '360DigiTMG', 2019, 2022)

# Creating another tuple named tup2
tup2 = (1, 2, 3, 4, 5, 6, 7)

# Printing the contents of tup1 and tup2
print(tup1)
print(tup2)

# Printing the first element of tup1 and a slice of tup2
print(tup1[0])
print(tup2[1:5])

# Accessing an element of tuple (tup1[1]) # Commented out since it causes an error
# Deletion of individual items of tuple (tup1[1]) # Commented out since tuples are immutable and this operation is not allowed.


# Create a SQLAlchemy engine to connect to the MySQL database
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user="root", pw=quote("Sai@123kumar"), db="association_rules"))



# Read the CSV file containing grocery data into a pandas DataFrame
data = pd.read_csv(r"groceries.csv", sep=';', header=None)

# Display the first few rows of the DataFrame
data.head()

# Write the data from the DataFrame to the MySQL database table named 'groceries'
data.to_sql('groceries', con=engine, if_exists='replace', chunksize=1000, index=False)

# Read data from the 'groceries' table in the database into a pandas DataFrame
sql = 'select * from groceries;'
groceries = pd.read_sql_query(sql, con=engine)

# Display the first few rows of the DataFrame
groceries.head()

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Convert the DataFrame 'groceries' into a list
groceries = groceries.iloc[:, 0].to_list()

# Display the list of groceries
groceries

# Extract items from the transactions (each transaction is represented as a string)
groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))

# Print the list of transactions after splitting them into individual items
print(groceries_list)

# Remove null values from the list of transactions
groceries_list_new = []
for i in groceries_list:
    groceries_list_new.append(list(filter(None, i)))

# Print the updated list of transactions without null values
print(groceries_list_new)

# TransactionEncoder: Encoder for transaction data in Python lists
# Encodes transaction data in the form of a Python list of lists, into a NumPy array
TE = TransactionEncoder()

# Fit the TransactionEncoder to the list of transactions
X_1hot_fit = TE.fit(groceries_list)

# Serialize the fitted TransactionEncoder using pickle and save it to a file named 'TE.pkl'
import pickle
pickle.dump(X_1hot_fit, open('TE.pkl', 'wb'))

# Get the current working directory
import os
os.getcwd()

# Load the fitted TransactionEncoder from the saved pickle file
X_1hot_fit1 = pickle.load(open('TE.pkl', 'rb'))

# Transform the list of transactions using the fitted TransactionEncoder
X_1hot = X_1hot_fit1.transform(groceries_list)

# Print the transformed transactions
print(X_1hot)

# Convert the transformed transactions into a DataFrame with columns as item names
transf_df = pd.DataFrame(X_1hot, columns=X_1hot_fit1.columns_)

# Display the transformed DataFrame
transf_df

# Get the shape of the transformed DataFrame
transf_df.shape

### Elementary Analysis ###

# Calculate the count of each item across all transactions
count = transf_df.loc[:, :].sum()

# Get the most popular items by sorting the counts in descending order and selecting the top 10
pop_item = count.sort_values(ascending=False).head(10)

# Convert the series of popular items into a DataFrame
pop_item = pop_item.to_frame()  # Type casting the series to DataFrame

# Reset the index of the DataFrame
pop_item = pop_item.reset_index()

# Rename the columns of the DataFrame
pop_item = pop_item.rename(columns={"index": "items", 0: "count"})

# Display the DataFrame of popular items
pop_item

# Data Visualization
# get_ipython().run_line_magic('matplotlib', 'inline')
# Set the runtime configuration for figure size
plt.rcParams['figure.figsize'] = (10, 6)  # rc stands for runtime configuration

# Set the style of the plot to dark background
plt.style.use('dark_background')

# Plot a horizontal bar chart for the most popular items
pop_item.plot.barh()

# Set the title of the plot
plt.title('Most popular items')

# Invert the y-axis to display items with the highest count at the top
plt.gca().invert_yaxis()  # gca means "get current axes"

# Display help documentation for the apriori function
help(apriori)

# Find frequent itemsets using the Apriori algorithm
frequent_itemsets = apriori(transf_df, min_support=0.0075, max_len=4, use_colnames=True)

# Display the frequent itemsets
frequent_itemsets

# Sort the frequent itemsets based on support in descending order
frequent_itemsets.sort_values('support', ascending=False, inplace=True)

# Display the frequent itemsets sorted by support
frequent_itemsets

# Generate association rules from the frequent itemsets using the lift metric
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Display the first 10 association rules
rules.head(10)

# Sort the association rules based on lift in descending order and display the top 10 rules
rules.sort_values('lift', ascending=False).head(10)


# Define a function to convert a set to a sorted list
def to_list(i):
    return (sorted(list(i)))

# Apply the function to sort the items in both Antecedents and Consequents of the rules, and merge them together
ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

# Sort the merged list of items (transactions)
ma_X = ma_X.apply(sorted)

# Convert the merged list of transactions into a list of lists
rules_sets = list(ma_X)

# Remove duplicates by converting the list of lists into a set of tuples (each tuple representing a unique transaction), and then converting it back to a list of lists
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

# Create an empty list to capture the index of unique item sets
index_rules = []

# Iterate over unique item sets and capture their indexes in the original list of rules_sets
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# Display the list of indexes of unique item sets
index_rules

# Filter the rules DataFrame to include only the rules without redundancy based on the captured indexes
rules_no_redundancy = rules.iloc[index_rules, :]

# Display the rules DataFrame without redundancy
rules_no_redundancy

# Sort the rules based on lift in descending order and select the top 10 rules
rules10 = rules_no_redundancy.sort_values('lift', ascending=False).head(10)

# Display the top 10 rules
rules10

# Plot a scatter plot for the top 10 rules, where support and confidence are represented by x and y axes respectively, and lift is represented by color
rules10.plot(x="support", y="confidence", c=rules10.lift, 
             kind="scatter", s=12, cmap=plt.cm.coolwarm)


# Store the rules on to SQL database
# Database do not accepting frozensets

# Convert 'antecedents' and 'consequents' columns to string type to facilitate string operations
rules10['antecedents'] = rules10['antecedents'].astype('string')
rules10['consequents'] = rules10['consequents'].astype('string')

# Remove the prefix "frozenset({" from the 'antecedents' column
rules10['antecedents'] = rules10['antecedents'].str.removeprefix("frozenset({")

# Remove the suffix "})" from the 'antecedents' column
rules10['antecedents'] = rules10['antecedents'].str.removesuffix("})")

# Remove the prefix "frozenset({" from the 'consequents' column
rules10['consequents'] = rules10['consequents'].str.removeprefix("frozenset({")

# Remove the suffix "})" from the 'consequents' column
rules10['consequents'] = rules10['consequents'].str.removesuffix("})")

# Write the modified DataFrame 'rules10' to the 'groceries_ar' table in the database
rules10.to_sql('groceries_ar', con=engine, if_exists='replace', chunksize=1000, index=False)


