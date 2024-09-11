# Import necessary modules from Flask
from flask import Flask, render_template, request

# Import pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# Import pickle for loading pickled objects
import pickle

# Import functions from mlxtend library for frequent itemsets and association rules
from mlxtend.frequent_patterns import apriori, association_rules

# Import create_engine function from SQLAlchemy for database connection
from sqlalchemy import create_engine
from urllib.parse import quote
# Load the fitted TransactionEncoder from the saved pickle file
X_1hot_fit1 = pickle.load(open('TE.pkl', 'rb'))

# Define the connection string for the SQLAlchemy engine
# This connects to a MySQL database named 'retail_db' with username 'user1' and password 'user1'



# Create a Flask application instance
app = Flask(__name__)

# Define the route for the home page
@app.route('/')
def home():
    # Render the index.html template
    return render_template('index.html')

# Define the route for the form submission
@app.route('/success', methods=['POST'])
def success():
    # Check if the request method is POST
    if request.method == 'POST':
        # Get the uploaded file from the request
        f = request.files['file']
        user = request.form['user']
        pw = quote(request.form['pw'])
        db = request.form['db']
        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(f, sep=';', header=None)
        
        # Creating database engine to connect to MySQL database
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
        # Convert the DataFrame to a list of transactions
        data = data.iloc[:, 0].to_list()
        
        # Split the transactions into individual items
        groceries_list = []
        for i in data:
           groceries_list.append(i.split(","))
        
        # Remove null values from the list of transactions
        groceries_list_new = []
        for i in groceries_list:
            groceries_list_new.append(list(filter(None, i)))
        
        # Transform the list of transactions using the fitted TransactionEncoder
        transf_df = pd.DataFrame(X_1hot_fit1.transform(groceries_list_new), columns=X_1hot_fit1.columns_)
        
        # Drop the first column (empty column) from the transformed DataFrame
        transf_df = transf_df.drop(transf_df.columns[0], axis=1)
        
        # Find frequent itemsets using Apriori algorithm
        frequent_itemsets = apriori(transf_df, min_support=0.0075, max_len=4, use_colnames=True)
        
        # Sort frequent itemsets based on support in descending order
        frequent_itemsets.sort_values('support', ascending=False, inplace=True)
        
        # Generate association rules from the frequent itemsets using the lift metric
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        
        # Define a function to convert a set to a sorted list
        def to_list(i):
            return sorted(list(i))
        
        # Merge Antecedents and Consequents and remove duplicates
        ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)
        ma_X = ma_X.apply(sorted)
        rules_sets = list(ma_X)
        unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
        
        # Capture the indexes of unique item sets
        index_rules = []
        for i in unique_rules_sets:
            index_rules.append(rules_sets.index(i))
        
        # Filter the rules DataFrame to include only the rules without redundancy
        rules_no_redundancy = rules.iloc[index_rules, :]
        
        # Sort the rules based on lift in descending order and select the top 15 rules
        rules_new = rules_no_redundancy.sort_values('lift', ascending=False).head(15)
        
        # Replace infinite values with NaN
        rules_new = rules_new.replace([np.inf, -np.inf], np.nan)
        
        # Convert 'antecedents' and 'consequents' columns to string type
        rules_new['antecedents'] = rules_new['antecedents'].astype('string')
        rules_new['consequents'] = rules_new['consequents'].astype('string')
        
        # Remove 'frozenset({})' from 'antecedents' and 'consequents' columns
        rules_new['antecedents'] = rules_new['antecedents'].str.removeprefix("frozenset({")
        rules_new['antecedents'] = rules_new['antecedents'].str.removesuffix("})")
        rules_new['consequents'] = rules_new['consequents'].str.removeprefix("frozenset({")
        rules_new['consequents'] = rules_new['consequents'].str.removesuffix("})")
        
        # Write the modified DataFrame to the MySQL database table 'groceries_ar'
        rules_new.to_sql('groceries_ar', con=engine, if_exists='replace', chunksize=1000, index=False)
        
        # Convert the DataFrame to an HTML table with Bootstrap styling
        html_table = rules_new.to_html(classes='table table-striped')
        
        # Render the new.html template with the HTML table
        return render_template("data.html", Y=f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #39648f;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                    .table td {{\
                        background-color: #5e617d;\
                    }}\
                    .table tbody th {{\
                        background-color: #ab2c3f;\
                    }}\
                </style>\
                {html_table}") 

# Run the Flask application if this script is executed
if __name__ == '__main__':
    app.run(debug=False)
