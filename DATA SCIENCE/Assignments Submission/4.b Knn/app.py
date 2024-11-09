from flask import Flask, render_template, request
import pandas as pd
import pickle
import joblib
from sqlalchemy import create_engine
from urllib.parse import quote 

model = pickle.load(open('knn.pkl','rb'))
imp_scale = joblib.load('imp_scale')
winsor = joblib.load('winsor')
# connecting to sql by creating sqlachemy engine
user_name = 'root'
database = 'knn'
your_password = 'Sai@123kumar'
engine = create_engine(f'mysql+pymysql://{user_name}:%s@localhost:3306/{database}' % quote(f'{your_password}'))

#define flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        data = pd.read_excel(f)
        
        cleaned = pd.DataFrame(imp_scale.transform(data), columns = imp_scale.get_feature_names_out())
        cleaned[list(cleaned.iloc[:,0:7])]  = winsor.transform(cleaned[list(cleaned.iloc[:,0:7])] )
        predictions = pd.DataFrame(model.predict(cleaned), columns = ['Type'])
        final = pd.concat([predictions, data], axis = 1)
        html_table = final.to_html(classes='table table-striped')


        final.to_sql('glass_predictions', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        return render_template("new.html", Y = f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #8f6b39;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                        .table td {{\
                        background-color: #32b8b8;\
                    }}\
                            .table tbody th {{\
                            background-color: #3f398f;\
                        }}\
                </style>\
                {html_table}") 
if __name__=='__main__':
    app.run(debug = False)
