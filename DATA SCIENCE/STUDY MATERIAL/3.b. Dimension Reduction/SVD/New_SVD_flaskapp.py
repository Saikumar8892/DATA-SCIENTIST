# Importing necessary modules
# from cProfile import run  # Importing cProfile module for performance profiling (commented out)
from flask import Flask, render_template, request  # Importing Flask module for creating web applications
from sqlalchemy import create_engine  # Importing create_engine from SQLAlchemy for database connection
import pandas as pd  # Importing Pandas library for data manipulation and analysis
from urllib.parse import quote
# Creating a Flask application instance
app = Flask(__name__)

# Importing the saved SVD model using joblib
import joblib
model = joblib.load("svd_DimRed")


# Route for the home page of the Flask application
@app.route('/')
def home():
    # Rendering the 'index.html' template when the root URL is accessed
    return render_template('index.html')

# Route for handling file upload and performing prediction
@app.route('/success', methods=['POST'])
def success():
    # Checking if the request method is POST
    # Checking if the request method is POST, indicating form submission
    if request.method == 'POST':
        
        f = request.files['file'] # Retrieving the uploaded file 
        user = request.form['user'] # Retrieving the username
        pw = quote(request.form['pw']) # Retrieving the password 
        db = request.form['db'] # Retrieving the database name

        # Creating an engine to connect to the MySQL database
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

        try:
            # Attempting to read the uploaded file as CSV
            data = pd.read_csv(f)
        except:
            try:
                # Attempting to read the uploaded file as Excel
                data = pd.read_excel(f)
            except:      
                # If unable to read as CSV or Excel, treating the file as DataFrame directly
                data = pd.DataFrame(f)
                  
        # Drop the unwanted feature 'UnivID'
        data1 = data.drop(["UnivID"], axis=1)
        
        # Selecting only numeric columns
        num_cols = data1.select_dtypes(exclude=['object']).columns
        
        # Performing dimensionality reduction (SVD) using the saved model
        svd_res = pd.DataFrame(model.transform(data1[num_cols]), columns=['svd0', 'svd1', 'svd2', 'svd3', 'svd4'])
        
        # Concatenating the university names and SVD components into a final dataframe
        final = pd.concat([data.Univ, svd_res], axis=1)
        
        # Writing the final dataframe to a new table 'university_pred_svd' in the database
        final.to_sql('university_pred_svd', con=engine, if_exists='replace', chunksize=1000, index=False)
        
        # Converting the final dataframe to HTML table format with specified classes for styling
        html_table = final.to_html(classes='table table-striped')
        
        # Rendering the 'data.html' template with the HTML table data
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
                        background-color: #a8dfe3;\
                    }}\
                            .table tbody th {{\
                            background-color: #ab2c3f;\
                        }}\
                </style>\
                {html_table}")

# Running the Flask application
if __name__ == '__main__':
    # Enabling debug mode for easier development
    app.run(debug=False)
