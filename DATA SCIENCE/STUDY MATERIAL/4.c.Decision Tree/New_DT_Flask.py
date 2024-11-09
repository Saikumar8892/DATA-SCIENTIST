# Importing necessary libraries
from flask import Flask, render_template, request  # For creating web applications using Flask
import re  # For regular expression operations
import pandas as pd  # For data manipulation and analysis
import copy  # For creating deep copies of objects
import pickle  # For serializing and deserializing Python objects
import joblib  # For saving and loading objects using joblib

# Load the trained Decision Tree model from the 'DT.pkl' file using pickle
model = pickle.load(open('DT.pkl', 'rb'))

# Load 'ise' pipeline stands for imputation, scaling and encoding using joblib
ise = joblib.load('ise')

# Load the winsorizer transformer from the 'winsor' file using joblib
winsor = joblib.load('winsor')

# Connecting to an SQL database using SQLAlchemy engine
from sqlalchemy import create_engine
from urllib.parse import quote

# Function to preprocess data and make predictions using the decision tree model
def decision_tree(data_new):
    # Apply preprocessing steps

    # Impute missing values using the mean imputer and transform the data
    clean1 = pd.DataFrame(ise.transform(data_new), columns=ise.get_feature_names_out())

    # Apply winsorization to specific numerical features (months_loan_duration, amount, age)
    clean1[['numeric__months_loan_duration', 'numeric__amount','numeric__age']] = winsor.transform(clean1[['numeric__months_loan_duration', 'numeric__amount','numeric__age']])
    
    # Make predictions using the preprocessed data
    prediction = pd.DataFrame(model.predict(clean1), columns=['default'])
    
    # Concatenate the predicted labels with the original data
    final_data = pd.concat([prediction, data_new], axis=1)
    
    # Return the final dataset with predicted labels
    return final_data
            
# Define Flask app
app = Flask(__name__)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')  # Renders the 'index.html' template for the home page

# Define route for success page
@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']  # Get the uploaded file from the request
        data_new = pd.read_csv(f)  # Read the uploaded data as a DataFrame
        user = request.form['user']  # Getting the database username from the form
        pw = quote(request.form['pw'])  # Getting the database password from the form
        db = request.form['db']  # Getting the database name from the form
        # Creating an engine to connect to the PostgreSQL database
        conn_string = f"postgresql+psycopg2://{user}:{pw}@localhost/{db}"
                       
        db = create_engine(conn_string)
        conn = db.connect()       
        # Make predictions and save to SQL database
        final_data = decision_tree(data_new)  # Make predictions using the decision_tree function
        final_data.to_sql('credit_test', con=conn, if_exists='replace', chunksize=1000, index=False)  # Save the final_data to SQL database
        html_data = final_data.to_html(classes='table table-striped')
        # Render a new HTML page displaying the predictions
        return render_template("data.html", Y= f"<style>\
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
                        background-color: #888a9e;\
                    }}\
                            .table tbody th {{\
                            background-color: #ab2c3f;\
                        }}\
                </style>\
                {html_data}")  # Renders the 'new.html' template with the predictions displayed as an HTML table

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode
