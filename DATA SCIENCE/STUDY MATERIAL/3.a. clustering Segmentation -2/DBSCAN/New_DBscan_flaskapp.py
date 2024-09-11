# Importing necessary libraries
from flask import Flask, render_template, request  # Flask for creating web application, render_template for rendering HTML templates, request for handling HTTP requests
from sqlalchemy import create_engine  # For creating a connection to the database
import pandas as pd  # For data manipulation and analysis
from urllib.parse import quote
# Importing the DBSCAN clustering model from a pickled file
import pickle
model = pickle.load(open('db.pkl', 'rb'))  # Loading the DBSCAN clustering model

# Creating a Flask web application instance
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Rendering the index.html template

# Route for the success page, which handles file upload and performs DBSCAN clustering
@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':  # Checking if the request method is POST
        f = request.files['file']  # Getting the uploaded file from the request
        user = request.form['user']  # Getting the username from the form
        pw = quote(request.form['pw'])  # Getting the password from the form
        db = request.form['db']  # Getting the database name from the form
        
        # Creating a database engine to connect to the MySQL database using the provided credentials
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
        
        try:
            data = pd.read_csv(f)  # Trying to read the uploaded file as a CSV file
        except:
            try:
                data = pd.read_excel(f)  # Trying to read the uploaded file as an Excel file
            except:      
                data = pd.DataFrame(f)  # If neither CSV nor Excel, creating a DataFrame from the file
            
        # Performing DBSCAN clustering on the data and adding the cluster labels as a new column
        prediction = pd.DataFrame(model.fit_predict(data), columns=['clusters'])
        prediction = pd.concat([prediction, data], axis=1)  # Concatenating the cluster labels with the original data
        
        # Saving the clustered data to a SQL table in the database
        prediction.to_sql('db_scan', con=engine, if_exists='replace', chunksize=1000, index=False)
        
        # Converting the clustered data DataFrame to an HTML table
        html_table = prediction.to_html(classes='table table-striped')
        
        # Rendering the data.html template with the HTML table
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
                        background-color: #e7e8bc;\
                    }}\
                            .table tbody th {{\
                            background-color: #ab2c3f;\
                        }}\
                </style>\
                {html_table}")

# Running the Flask application
if __name__ == '__main__':
    app.run(debug=False)  # Running the Flask app in debug mode
