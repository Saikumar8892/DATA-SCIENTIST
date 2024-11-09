# Import libraries
import pandas as pd  # For data manipulation
import streamlit as st  # For building interactive web apps

import pickle  # For loading saved machine learning model
from PIL import Image  # For displaying images

# Load the saved model
model = pickle.load(open('stacking_iris.pkl', 'rb'))

def predict(data):
    """
    Function to make predictions using the loaded model.

    Args:
    data (DataFrame): Input data for prediction.

    Returns:
    DataFrame: DataFrame containing predictions.
    """

    prediction = pd.DataFrame(model.predict(data), columns=['species'])
    prediction = pd.concat([prediction, data], axis=1)
    
    return prediction

def main():  
    st.set_page_config(layout="wide")

    st.title("Stacking_Classifier")
    st.sidebar.title("Stacking_Classifier")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Stacking_Classifier</h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("")
    
    # Allow user to upload a file
    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'], accept_multiple_files=False, key="fileUploader")
    if uploadedFile is not None:
        try:
            data = pd.read_csv(uploadedFile)
        except:
            try:
                data = pd.read_excel(uploadedFile)
            except:
                data = pd.DataFrame(uploadedFile)
    else:
        st.sidebar.warning("You need to upload a csv or excel file.")
    
    # Make predictions when the Predict button is clicked
    if st.button("Predict"):
        result = predict(data)
                                   
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap=True)
        st.table(result.style.background_gradient(cmap=cm))

if __name__=='__main__':
    main()
