# Import necessary libraries
from PIL import Image

import pandas as pd  # Import pandas library for data manipulation
import numpy as np  # Import numpy library for numerical operations
import streamlit as st  # Import streamlit for creating web applications
from sqlalchemy import create_engine  # Import create_engine from SQLAlchemy for database connection
import joblib, pickle  # Import joblib and pickle for loading saved models
from urllib.parse import quote
# Load the logistic regression model and preprocessing objects
model1 = pickle.load(open('logistic.pkl', 'rb'))  # Load the logistic regression model using pickle
clean = joblib.load('clean')  # Load the  preprocessing object using joblib

def predict_MPG(data, user, pw, db):
    """
    Predicts attorney involvement (ATTORNEY) based on given data,
    preprocesses it using saved pipelines, makes predictions with a
    loaded model, applies a threshold, and stores results in a database.

    Args:
        data (pd.DataFrame): Input DataFrame containing prediction features.
        user (str): Username for database connection.
        pw (str): Password for database connection.
        db (str): Database name.

    Returns:
        pd.DataFrame: DataFrame with predictions and actual values.
    """

    # Establish a database connection for storing predictions
    engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))
    
    # Remove the 'CASENUM' column as it's not used for prediction
    data = data.drop('CASENUM', axis=1)


    clean1 = pd.DataFrame(clean.transform(data), columns = data.columns)

    # Generate predictions using the trained logistic regression model
    prediction = model1.predict(clean1)
    
    # Define the optimal threshold for classifying attorney involvement
    optimal_threshold = 0.6027403450992425  # Placeholder, replace with actual value
    
    # Create a new column named "ATTORNEY" to store predicted labels (initially all zeros)
    data["ATTORNEY"] = np.zeros(len(prediction))
    
    # Apply the threshold to classify observations based on predicted probabilities
    data.loc[prediction > optimal_threshold, "ATTORNEY"] = 1  # Set "ATTORNEY" to 1 for predictions exceeding the threshold
    
    # Ensure "ATTORNEY" column is in integer format for compatibility
    data[['ATTORNEY']] = data[['ATTORNEY']].astype('int64')


    # Save predicted data to the database
    data.to_sql('attorney_predictions', con=engine, if_exists='replace', chunksize=1000, index=False)

    return data

def main():
    image = Image.open("AiSPRY logo.jpg") # Opening an image file named "AiSPRY logo" using the Image.open() function
    st.sidebar.image(image) # Displaying the image in the sidebar of a Streamlit application using the st.sidebar.image() function

    # Streamlit app title displayed on the main page
    st.title("Attorney for Claims Cases Prediction")

    # Streamlit sidebar title displayed on the left-hand panel
    st.sidebar.title("Attorney for Claims Cases Prediction")

    # Define HTML template for custom styling
    html_temp = """
    <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Attorney for Claims Cases Prediction App</h2>
    </div>
    """

    # Allow file upload (CSV or Excel) in the sidebar for user interaction
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv','xlsx'], accept_multiple_files=False, key="fileUploader")

    # Handle file upload and format detection for robustness
    if uploaded_file is not None:
        try:
            # Read uploaded CSV file if it's a CSV format
            data = pd.read_csv(uploaded_file)
        except:
            try:
                # Read uploaded Excel file if CSV fails (fallback option)
                data = pd.read_excel(uploaded_file)
            except:
                # Set data to an empty DataFrame if both formats fail
                data = pd.DataFrame()
    else:
        # Display warning if no file is uploaded for user feedback
        st.sidebar.warning("You need to upload a CSV or an Excel file.")

    # HTML template for styling the section title
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add Database Credentials </p>
    </div>
    """
    # Display the styled section title in the sidebar
    st.sidebar.markdown(html_temp, unsafe_allow_html=True)

    # Input fields for database credentials in the sidebar for user input
    # Create a text input field for the user's database username
    user = st.sidebar.text_input("user", "Type Here")
    
    # Create a text input field for the user's database password
    pw = st.sidebar.text_input("password", "Type Here", type="password")  # Use type="password" for masking input
    
    # Create a text input field for the user's database name
    db = st.sidebar.text_input("database", "Type Here")

    # Initialize an empty variable to store results for later use
    result = ""

    # Button to trigger prediction with clear action text
    if st.button("Predict"):
        # Call the predict_MPG function with user input and display results
        result = predict_MPG(data, user, pw, db)
        # Import seaborn for table styling (assuming seaborn is installed)
        import seaborn as sns
        # Create a color palette for table background gradient
        cm = sns.light_palette("blue", as_cmap=True)
        # Display results in a table with background gradient
        st.table(result.style.background_gradient(cmap=cm))

# Run the main function when the script is executed directly
if __name__ == '__main__':
    main()
