# Importing necessary libraries
from flask import Flask, render_template, request  # For creating web applications using Flask
import re  # For regular expression operations
import pandas as pd  # For data manipulation and analysis
import pickle  # For serializing and deserializing Python objects (such as machine learning models)
import joblib  # For saving and loading objects using joblib

# Load preprocessing transformers and trained models
ise = joblib.load('ise')           # Mean imputer # Min-Max Scaler # One-Hot Encoder
winsor = joblib.load('winsor')               # Winsorizer
bagging = pickle.load(open('baggingmodel.pkl', 'rb'))            # Bagging model
rfc = pickle.load(open('rfc.pkl', 'rb'))                          # Random Forest Classifier
adaboost = pickle.load(open('adaboost.pkl', 'rb'))                # AdaBoost model
gradiantboost = pickle.load(open('gradiantboostparam.pkl', 'rb')) # Gradient Boosting model
xgboost = pickle.load(open('Randomizedsearch_xgb.pkl', 'rb'))     # XGBoost model

# Connecting to SQL by creating a sqlalchemy engine
from sqlalchemy import create_engine
from urllib.parse import quote

# Define Flask app
app = Flask(__name__)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')  # Renders the 'index.html' template for the home page

# Define route for success page
@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':  # Checks if the request method is POST
        f = request.files['file']          # Get uploaded file
        
        user = request.form['user']
        pw = quote(request.form['pw'])
        db = request.form['db']
        try:
            # Reading the uploaded file (either CSV or Excel)
            data = pd.read_csv(f)
        except:
            try:
                data = pd.read_excel(f)
            except:
                data = pd.DataFrame(f)
        # Creating database engine to connect to MySQL database
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
           
        clean_data = pd.DataFrame(ise.transform(data), columns=ise.get_feature_names_out())
        clean_data[list(clean_data.iloc[:,:16].columns)] = winsor.transform(clean_data[list(clean_data.iloc[:,:16].columns)])   # Apply winsorization
        # Make predictions using each model
        prediction = pd.DataFrame(bagging.predict(clean_data), columns=['Bagging_Oscar'])
        prediction1 = pd.DataFrame(rfc.predict(clean_data), columns=['RFC_Oscar'])
        prediction2 = pd.DataFrame(adaboost.predict(clean_data), columns=['Adaboost_Oscar'])
        prediction3 = pd.DataFrame(gradiantboost.predict(clean_data), columns=['Gradientboost_Oscar'])
        prediction4 = pd.DataFrame(xgboost.predict(clean_data), columns=['XGboost_Oscar'])
        # Concatenate predictions with original data
        final_data = pd.concat([prediction, prediction1, prediction2, prediction3, prediction4, data], axis=1)
        # Save predictions to SQL database
        final_data.to_sql('bagging_test', con=engine, if_exists='replace', chunksize=1000, index=False)
        # Render a new HTML page displaying the predictions
        html_table = final_data.to_html(classes='table table-striped')
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

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app
