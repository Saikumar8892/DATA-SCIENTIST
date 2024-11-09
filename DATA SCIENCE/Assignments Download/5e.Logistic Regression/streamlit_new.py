# Import necessary libraries
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine
import joblib, pickle
from urllib.parse import quote

# Load the logistic regression model and preprocessing objects
model1 = pickle.load(open('logistic_model.pkl', 'rb'))  # Load the logistic regression model
clean = joblib.load('preprocess_pipeline.pkl')  # Load the preprocessing pipeline

def predict_clicks(data, user, pw, db):
    """
    Predicts if a user will click on an ad based on input data.
    Preprocesses data using saved pipelines, makes predictions,
    applies a threshold, and stores results in a database.

    Args:
        data (pd.DataFrame): Input DataFrame containing prediction features.
        user (str): Username for database connection.
        pw (str): Password for database connection.
        db (str): Database name.

    Returns:
        pd.DataFrame: DataFrame with predictions.
    """
    
    # Establish a database connection
    engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))
    
    # Drop irrelevant columns for prediction if present
    data = data.drop(['User ID'], axis=1, errors='ignore')
    
    # Preprocess the data
    clean_data = pd.DataFrame(clean.transform(data), columns=data.columns)
    
    # Generate predictions
    prediction = model1.predict(clean_data)
    
    # Set optimal threshold
    optimal_threshold = 0.5  # Replace with the best threshold determined from training
    
    # Add "Clicked on Ad" column based on predictions
    data["Clicked on Ad"] = (prediction > optimal_threshold).astype(int)
    
    # Store predictions in the database
    data.to_sql('ad_click_predictions', con=engine, if_exists='replace', index=False)
    
    return data

def main():
    # Load and display an image in the sidebar
    image = Image.open("AiSPRY_logo.jpg")
    st.sidebar.image(image)
    
    # Title for the Streamlit app
    st.title("Ad Click Prediction App")

    # Sidebar for file upload and database credentials
    st.sidebar.title("Upload Data & Database Credentials")

    # HTML styling for sidebar titles
    html_temp = """
    <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Predict Ad Clicks</h2>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'])

    # Load data if a file is uploaded
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception:
            data = pd.read_excel(uploaded_file)

    # Database credentials
    user = st.sidebar.text_input("Database User", "Type Here")
    pw = st.sidebar.text_input("Database Password", "Type Here", type="password")
    db = st.sidebar.text_input("Database Name", "Type Here")

    # Button to initiate prediction
    if st.button("Predict"):
        if uploaded_file is not None and user and pw and db:
            result = predict_clicks(data, user, pw, db)
            st.write("Predictions:")
            st.dataframe(result.style.background_gradient(cmap="Blues"))
        else:
            st.sidebar.warning("Please upload data and enter database credentials.")

# Run the app
if __name__ == '__main__':
    main()

