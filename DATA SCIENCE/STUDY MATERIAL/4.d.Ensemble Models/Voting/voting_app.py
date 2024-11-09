# Import libraries
import pandas as pd  # For data manipulation
import streamlit as st  # For building interactive web apps

import pickle  # For loading saved machine learning models
from PIL import Image  # For working with images

# Load the saved models
model1 = pickle.load(open('hard_voting.pkl', 'rb'))  # Load the hard voting model
model2 = pickle.load(open('soft_voting.pkl', 'rb'))  # Load the soft voting model

def predict(data, model):
    """
    Function to make predictions using the specified model.

    Args:
    data (DataFrame): Input data for prediction.
    model: Voting classifier model for prediction.

    Returns:
    DataFrame: DataFrame containing predictions.
    """
    prediction = pd.DataFrame(model.predict(data), columns=['target'])  # Make predictions using the model
    prediction = pd.concat([prediction, data], axis=1)  # Concatenate predictions with input data
    return prediction

def main():  
    st.set_page_config(layout="wide")

    st.title("Voting")
    st.sidebar.title("Voting")

    # HTML template for title
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Stacking_Regression</h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html=True)  # Display title

    st.text("")

    # Allow user to upload a file
    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'], accept_multiple_files=False, key="fileUploader")
    if uploadedFile is not None:
        try:
            data = pd.read_csv(uploadedFile)  # Read uploaded CSV file
        except:
            try:
                data = pd.read_excel(uploadedFile)  # Read uploaded Excel file
            except:
                data = pd.DataFrame(uploadedFile)  # If neither CSV nor Excel, treat as DataFrame
    else:
        st.sidebar.warning("You need to upload a csv or excel file.")  # Display warning if no file uploaded
    
    # Select voting type
    selected_option = st.selectbox('Select a Voting type', ['Hard Voting', 'Soft Voting'])
    if selected_option == "Hard Voting":
        model = model1  # Use hard voting model
    else:
        model = model2  # Use soft voting model
    
    # Make predictions when the Predict button is clicked
    if st.button("Predict"):
        result = predict(data, model)  # Predict using selected model
                                   
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap=True)
        st.table(result.style.background_gradient(cmap=cm))  # Display predictions in a styled table

if __name__=='__main__':
    main()
