# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
from PIL import Image # Importing the Image class from the Python Imaging Library (PIL)
import streamlit as st  # For creating interactive web apps

# SQLAlchemy is used to interact with SQL databases
from sqlalchemy import create_engine  
from urllib.parse import quote
import joblib  # For saving and loading models
import pickle  # For saving and loading objects in Python

# Load the pre-trained machine learning model from a pickle file
model1 = pickle.load(open('svc_rcv.pkl', 'rb'))

# Load the preprocessing objects from joblib files
clean = joblib.load('clean')  


# Define a function to predict Y (target) based on input data
def predict_Y(data, user, pw, db):
    """
    Predicts a target variable (Y) using a pre-trained model and stores results in a database.

    Args:
        data: A pandas DataFrame containing new data to be predicted on.
        user: The username for database connection.
        pw: The password for database connection.
        db: The name of the database to connect to.

    Returns:
        The final DataFrame with predictions and original data.
    """

    # Establish connection to MySQL database (requires credentials)
    engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))

    # Handle missing values in the input data using the pre-trained imputer
    clean1 = pd.DataFrame(clean.transform(data), columns=data.columns)

    
    # Make predictions on the preprocessed data using the pre-trained model
    prediction = pd.DataFrame(model1.predict(clean1), columns=['letter_pred'])

    # Combine the predictions with the original data for easier analysis
    final = pd.concat([prediction, data], axis=1)

    # Write the final DataFrame to the specified table in the SQL database
    final.to_sql('svm_test', con=engine, if_exists='replace', chunksize=1000, index=False)

    # Return the final DataFrame for further usage or visualization
    return final


# Define the main function for the Streamlit web app
def main():
    image = Image.open("AiSPRY logo.jpg") # Opening an image file named "AiSPRY logo" using the Image.open() function
    st.sidebar.image(image) # Displaying the image in the sidebar of a Streamlit application using the st.sidebar.image() function
    
    # Set the title for the web app
    st.title("Letter Classification")
    
    # Set the title for the sidebar
    st.sidebar.title("Letter Classification")
    
    # Define HTML template for styling
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Letter prediction </h2>
    </div>
    """
    
    # Render the HTML template
    # Render previously generated HTML content (presumably for a user interface element)
    st.markdown(html_temp, unsafe_allow_html=True)  # Display HTML content, enabling potential dynamic elements
    
    # Add a vertical space for visual separation
    st.text("")  # Create an empty line for better readability
    
    # Create a file uploader widget in the sidebar for user interaction
    uploadedFile = st.sidebar.file_uploader(
        "Upload a file",  # Label displayed to the user
        type=['csv', 'xlsx'],  # Allow only CSV and Excel files
        accept_multiple_files=False,  # Restrict to single file uploads
        key="fileUploader"  # Unique key to associate with the widget
    )

    # If a file is uploaded, attempt to read it as CSV or Excel
    if uploadedFile is not None:
        # Check if an uploaded file exists.
        
        try:
            data = pd.read_csv(uploadedFile)
            # Try to read the uploaded file as a CSV file using pandas.
            
        except:
            try:
                data = pd.read_excel(uploadedFile)
                # If reading as a CSV fails, try to read the uploaded file as an Excel file using pandas.
                
            except:      
                data = pd.DataFrame(uploadedFile)
                # If reading as both CSV and Excel fails, treat the uploaded file as a DataFrame directly.
                
    else:
        st.sidebar.warning("Upload a CSV or Excel file.")
        # If no file is uploaded, display a warning message in the Streamlit sidebar.

    
    # HTML template for adding database credentials
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credentials </p>
    </div>
    """
    
    # Render the HTML template
    st.sidebar.markdown(html_temp, unsafe_allow_html=True)
    # Display HTML content (specified by 'html_temp') in the Streamlit sidebar.
    # 'markdown' is used to interpret the HTML content.
    # 'unsafe_allow_html=True' allows rendering of HTML content, which could potentially be unsafe.
    
    # Input fields for database credentials
    user = st.sidebar.text_input("user", "Type Here")
    # Create a text input widget in the Streamlit sidebar for the user to input their username.
    # The default value displayed in the input field is "Type Here".
    
    pw = st.sidebar.text_input("password", "Type Here",type = "password")
    # Create a text input widget in the Streamlit sidebar for the user to input their password.
    # The default value displayed in the input field is "Type Here".
    
    db = st.sidebar.text_input("database", "Type Here")
    # Create a text input widget in the Streamlit sidebar for the user to input the database name.
    # The default value displayed in the input field is "Type Here".

    
    result = ""
    
    # If the "Predict" button is clicked, perform prediction
    if st.button("Predict"):
    # Check if the "Predict" button is clicked in the Streamlit app.
    # 'st.button' creates a button widget with the label "Predict".
    
        result = predict_Y(data, user, pw, db)
        # Call the 'predict_Y' function to generate predictions based on the provided data and database credentials.
        # Store the result in the variable 'result'.
        
        # Visualize the prediction results in a table with gradient background
        import seaborn as sns
        # Import the seaborn library for data visualization.
        
        cm = sns.light_palette("yellow", as_cmap=True)
        # Generate a color map using a light yellow palette from seaborn.
        # This color map will be used to style the table background gradient.
        
        st.table(result.style.background_gradient(cmap=cm))
        # Display the prediction results as a table in the Streamlit app.
        # Apply background gradient styling to the table using the specified color map.
        # Set the precision of numerical values in the table to 2 decimal places.

# Entry point of the script
if __name__ == '__main__':
    main()

