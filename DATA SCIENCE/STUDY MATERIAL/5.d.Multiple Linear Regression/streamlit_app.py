# Import necessary libraries
from PIL import Image

import pandas as pd  # Data manipulation and analysis
import streamlit as st  # Web application development framework
from sqlalchemy import create_engine  # Database connection management
import pickle, joblib  # Loading saved models and preprocessing objects
from urllib.parse import quote
# Load the pre-trained model and preprocessing objects
model1 = pickle.load(open('mpg.pkl', 'rb'))  # Load the model for fuel efficiency prediction
clean = joblib.load('preprocessed')  # Load the preprocessed pipeline for clean the data


# Define a function to predict fuel efficiency and save results to a database
def predict_MPG(data, user, pw, db):
    """
    Predicts fuel efficiency (MPG) for given car data, 
    saves predictions to a database, and returns the results.

    Args:
       data (pd.DataFrame): DataFrame containing car features.
       user (str): Username for database connection.
       pw (str): Password for database connection.
       db (str): Database name.

    Returns:
       pd.DataFrame: DataFrame with original data and predicted MPG.
    """

    # Create a SQLAlchemy engine for database interaction
    engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))

    # Preprocess the input data
    clean1 = pd.DataFrame(clean.transform(data), columns=clean.get_feature_names_out(input_features=data.columns)) 
    
    clean_data1 = clean1.drop('numerical__WT', axis=1)  # Drop potentially irrelevant column (adjust index as needed)
    
    # Generate predictions using the loaded model
    # Create a DataFrame of predictions using the model
    prediction = pd.DataFrame(model1.predict(clean_data1), columns=['Predicted_MPG'])
    
    # Combine the predictions with the original data horizontally
    final = pd.concat([prediction, data], axis=1)
    
    # Save the combined results to the database table 'mpg_predictions'
    final.to_sql('mpg_predictions', con=engine, if_exists='replace', chunksize=1000, index=False)

    return final  # Return the DataFrame with results


# Define the main function for the Streamlit application
def main():
    image = Image.open("AiSPRY logo.jpg") # Opening an image file named "AiSPRY logo" using the Image.open() function
    st.sidebar.image(image) # Displaying the image in the sidebar of a Streamlit application using the st.sidebar.image() function

    # Set the main title of the web application
    st.title("Fuel Efficiency Prediction")
    
    # Set the title for the sidebar for consistency and clarity
    st.sidebar.title("Fuel Efficiency Prediction")
    
    # Display a styled heading using HTML markup for visual appeal
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Cars Fuel Efficiency Prediction App </h2>
    </div>
    """
    # Render the styled heading in the app using markdown
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("")  # Add a spacer for better layout

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
