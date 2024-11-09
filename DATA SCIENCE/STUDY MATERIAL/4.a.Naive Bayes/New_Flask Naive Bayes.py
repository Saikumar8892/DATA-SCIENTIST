# Import necessary libraries
# Importing necessary libraries for Flask web application
from flask import Flask, render_template, request
# Importing libraries for data manipulation and processing
import pandas as pd
# Importing joblib for model serialization
import joblib
from sqlalchemy.exc import OperationalError
from sqlalchemy import create_engine
from urllib.parse import quote


# Loading the saved model into memory
model = joblib.load('processed1')

# Defining the Flask application
app = Flask(__name__)

# Defining a route for the home page of the web application
@app.route('/')
def home():
    return render_template('index.html')

# Defining a route for the success page of the web application
@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        # Reading the uploaded file
        f = request.files['file']
        email_data = pd.read_excel(f)
        user = request.form['user']  # Getting the database username from the form
        pw = quote(request.form['pw'])  # Getting the database password from the form
        db = request.form['db']  # Getting the database name from the form
        # Creating an engine to connect to the PostgreSQL database
       # Attempt MySQL connection
        conn_string_mysql = f"mysql+pymysql://{user}:{pw}@localhost/{db}"
        try:
            engine = create_engine(conn_string_mysql)
            conn = engine.connect()
            print("MySQL Connection Successful")
        except OperationalError:
            # If MySQL connection fails, attempt PostgreSQL connection
            conn_string_postgresql = f"postgresql+psycopg2://{user}:{pw}@localhost/{db}"
            try:
                engine = create_engine(conn_string_postgresql)
                conn = engine.connect()
                print("PostgreSQL Connection Successful")
            except OperationalError:
                print("Both MySQL and PostgreSQL connections failed. Unable to connect to the database.")
            
        # Making predictions using the loaded model
        test_pred_lap = pd.DataFrame(model.predict(email_data.text))
        test_pred_lap.columns = ["spam_pred"]

        # Concatenating original data with predicted values
        final = pd.concat([email_data, test_pred_lap], axis=1)
        
        # Saving predictions to the PostgreSQL database
        final.to_sql('sms_predictions', con=conn, if_exists='replace', index=False)
        
        html_table = final.to_html(classes='table table-striped')
        # Rendering a new HTML page displaying the predictions
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
                        background-color: #888a9e;\
                    }}\
                            .table tbody th {{\
                            background-color: #ab2c3f;\
                        }}\
                </style>\
                {html_table}")

# Running the Flask application
if __name__ == '__main__':
    app.run(debug=True)
