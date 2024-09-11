# Importing necessary libraries
from flask import Flask, render_template, request
from sqlalchemy import create_engine
from urllib.parse import quote
import pandas as pd
import joblib

# Initializing Flask application
app = Flask(__name__)

# Loading the PCA model
model = joblib.load("Data_prep_DimRed")

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for file upload and prediction
@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        # Getting file, user credentials, and database name from the form
        f = request.files['file']
        user = request.form['user']
        pw = quote(request.form['pw'])
        db = request.form['db']
        
        # Creating database engine to connect to MySQL database
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
        
        try:
            # Reading the uploaded file (either CSV or Excel)
            data = pd.read_csv(f)
        except:
            try:
                data = pd.read_excel(f)
            except:
                data = pd.DataFrame(f)
                
        # Dropping the unwanted feature (UnivID)
        data1 = data.drop(["UnivID"], axis=1)
        
        # Selecting only numeric columns
        num_cols = data1.select_dtypes(exclude=['object']).columns
        
        # Transforming the data using the saved PCA model
        pca_res = pd.DataFrame(model.transform(data1[num_cols]), columns=['pc0', 'pc1', 'pc2', 'pc3', 'pc4', 'pc5'])
        
        # Concatenating University names with the PCA results
        final = pd.concat([data.Univ, pca_res], axis=1)
        
        # Saving the results to the MySQL database
        final.to_sql('university_pred_pca', con=engine, if_exists='replace', chunksize=1000, index=False)
        
        # Creating HTML table for displaying the results on the webpage
        html_table = final.to_html(classes='table table-striped')
        
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
    app.run(debug=True)
