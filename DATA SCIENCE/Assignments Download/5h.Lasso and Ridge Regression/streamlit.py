import pandas as pd
import streamlit as st 
import numpy as np

import pickle, joblib
# Save the best model
from sqlalchemy import create_engine
from urllib.parse import quote

model1 = pickle.load(open('grid_lasso.pkl','rb'))
imp_enc_scale = joblib.load('imp_enc_scale')
winsor = joblib.load('winsor')



def predict_profit(data, user, pw, db):
    engine = create_engine(f'mysql+pymysql://{user}:%s@localhost:3306/{db}' % quote(f'{pw}'))
    clean = pd.DataFrame(imp_enc_scale.transform(data),columns= imp_enc_scale.get_feature_names_out())
    clean[list(clean.iloc[:,0:3])] = winsor.transform(clean[list(clean.iloc[:,0:3])])
    prediction = pd.DataFrame(model1.predict(clean), columns = ['Predict_profit'])
    final_data = pd.concat([prediction,data], axis = 1)
    final_data.to_sql('predict_profit', con = engine, if_exists='replace', chunksize=1000, index=False)
    return final_data


def main():
    

    st.title("Profit prediction")
    st.sidebar.title("Start up companies profit prediction")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Cars Fuel Efficiency Prediction App </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    
    uploadedFile = st.sidebar.file_uploader("Choose a file" ,type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
    if uploadedFile is not None :
        try:

            data=pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    df=pd.DataFrame()
        
        
    else:
        st.sidebar.warning("you need to upload a csv or excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = predict_profit(data, user, pw, db)
        #st.dataframe(result) or
        #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap=True)
        st.table(result.style.background_gradient(cmap=cm).set_precision(2))

                           
if __name__=='__main__':
    main()


