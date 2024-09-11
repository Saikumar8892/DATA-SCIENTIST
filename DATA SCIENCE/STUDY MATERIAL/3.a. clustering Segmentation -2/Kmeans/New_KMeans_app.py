# Importing necessary libraries
from flask import Flask, render_template, request  # For creating web application and handling HTTP requests
from sqlalchemy import create_engine  # For creating database connection
import pandas as pd  # For data manipulation
import pickle  # For loading KMeans clustering model
import joblib  # For loading Imputation and Scaling pipeline
from urllib.parse import quote
# Loading the Imputation and Scaling pipeline from a file
processed1 = joblib.load('preprocessing')

# Loading the KMeans clustering model from a file
model = pickle.load(open('clust_UNIV.pkl', 'rb'))

# Creating a Flask web application instance
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Rendering the index.html template

# Route for the success page
@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']  # Getting the uploaded file from the form
        user = request.form['user']  # Getting the database username from the form
        pw = quote(request.form['pw'])  # Getting the database password from the form
        db = request.form['db']  # Getting the database name from the form
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")  # Creating database connection
        
        try:
            data = pd.read_csv(f)  # Trying to read the uploaded file as a CSV
        except:
            try:
                data = pd.read_excel(f)  # Trying to read the uploaded file as an Excel file
            except:      
                data = pd.DataFrame(f)  # Creating a DataFrame from the uploaded file if it's not in CSV or Excel format
                  
        # Dropping the unwanted features from the data
        univ_df = data.drop(["UnivID", "Univ"], axis=1)

        # Selecting numeric features from the data
        #numeric_features = univ_df.select_dtypes(exclude=['object']).columns
        
        # Transforming the numeric features using the Imputation and Scaling pipeline
        data1 = pd.DataFrame(processed1.transform(univ_df), columns=processed1.get_feature_names_out())
        
        # Making predictions using the KMeans clustering model
        prediction = pd.DataFrame(model.predict(data1), columns=['cluster_id'])
        
        # Combining the predictions with the original data
        prediction = pd.concat([prediction, data], axis=1)
        
        # Writing the predictions to a database table
        prediction.to_sql('university_pred_kmeans', con=engine, if_exists='append', chunksize=1000, index=False)
        
        # Generating an HTML table from the predictions data
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
                        background-color: #888a9e;\
                    }}\
                            .table tbody th {{\
                            background-color: #ab2c3f;\
                        }}\
                </style>\
                {html_table}")

# Running the Flask application
if __name__ == '__main__':
    app.run(debug=False)  # Running the Flask app in debug mode
