# Import libraries for data manipulation, web app creation (if applicable), and database connection (if applicable)
import pandas as pd
from PIL import Image # Importing the Image class from the Python Imaging Library (PIL)
import streamlit as st  # For creating web applications (optional)
from sqlalchemy import create_engine  # For connecting to databases (optional)
from statsmodels.tools.tools import add_constant 
# Import libraries for model loading and preprocessing
import pickle
import joblib
from urllib.parse import quote
# Load the trained ElasticNet model for prediction
model1 = pickle.load(open('grid_elasticnet.pkl', 'rb'))  # Load best ElasticNet model
# Load the preprocessing pipelines used during model training
clean = joblib.load('clean') 

# Define a function to make predictions and store them in a database
def predict_MPG(data, user, pw, db):
    
    # Connect to the MySQL database
    engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))
    # Apply preprocessing to the input data
    # Preprocess numerical features
    clean_data = pd.DataFrame(clean.transform(data), columns = clean.get_feature_names_out())
    P = add_constant(clean_data)
    
    # Optionally, drop a feature with high VIF (replace '3' with actual feature name)
    clean_data1 = P.drop('numerical__WT', axis = 1)# Drop feature based on VIF analysis (example, adjust index accordingly)

    # Make predictions using the loaded model
    prediction = pd.DataFrame(model1.predict(clean_data1), columns=['Predict_MPG'])
    
    # Combine predictions with the input data
    final = pd.concat([prediction, data], axis=1)
    
    # Save the final DataFrame containing predictions and original data to a database table
    final.to_sql('mpg_predictions', con=engine, if_exists='replace', chunksize=1000, index=False)  # Save predictions to database table
    
    # Return the final DataFrame (optional)
    # You can uncomment this line to return the final DataFrame after saving it
    return final


# Define the main function to create the Streamlit app interface
def main():
    image = Image.open("AiSPRY logo.jpg") # Opening an image file named "AiSPRY logo" using the Image.open() function
    st.sidebar.image(image) # Displaying the image in the sidebar of a Streamlit application using the st.sidebar.image() function

    # Set the title for the app and sidebar
    st.title("Fuel Efficiency Prediction")
    st.sidebar.title("Fuel Efficiency Prediction")

    # Define HTML template for styling
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Cars Fuel Efficiency Prediction App </h2>
    </div>
    """
    # Display the HTML template
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("")
    
    # Create a file uploader for the user to upload data
    uploadedFile = st.sidebar.file_uploader("Choose a file" , type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
    

    # Check if a file has been uploaded using Streamlit
    if uploadedFile is not None:
        try:
            # Attempt to read the uploaded data as a CSV file
            data = pd.read_csv(uploadedFile)
            print("Successfully read data from CSV file.")  # Optional for debugging
        except:
            try:
                # If CSV fails, attempt to read as an Excel file
                data = pd.read_excel(uploadedFile)
                print("Successfully read data from Excel file.")  # Optional for debugging
            except:
                # If both CSV and Excel fail, create an empty DataFrame
                data = pd.DataFrame()
                print("Failed to read uploaded file.")  # Optional for debugging
    else:
        # If no file is uploaded, display a warning message in the Streamlit sidebar
        st.sidebar.warning("You need to upload a CSV or Excel file.")

    
    # HTML template for database credentials
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add Database Credentials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html=True)
            
    # Allow the user to input their username, password, and database name via text input fields in the sidebar.
    # 'st.sidebar.text_input' creates a text input widget in the Streamlit sidebar.
    # The first argument is the label displayed next to the input field.
    # The second argument is the default value displayed in the input field.
    user = st.sidebar.text_input("User", "Type Here")  # Input field for username
    pw = st.sidebar.text_input("Password", "Type Here",type = 'password')  # Input field for password
    db = st.sidebar.text_input("Database", "Type Here")  # Input field for database name
    
    result = ""  # Initialize a variable to store the result of user interaction or processing.
    # This variable will be used to display messages or outcomes to the user.

    # If the predict button is clicked, make predictions and display the result
    if st.button("Predict"):
    # Check if the "Predict" button is clicked.
    # 'st.button' creates a button widget in Streamlit with the specified label.
    # If the button is clicked, the following code block will be executed.

        result = predict_MPG(data, user, pw, db)
        # Call the 'predict_MPG' function to generate predictions based on the provided data and database credentials.
        # Assign the returned result to the 'result' variable.
    
        import seaborn as sns
        # Import the 'seaborn' library, often used for statistical data visualization.
    
        cm = sns.light_palette("blue", as_cmap=True)
        # Generate a color map using a light blue palette from seaborn.
        # This color map will be used to style the table background gradient.
    
        st.table(result.style.background_gradient(cmap=cm))
        # Display the prediction results as a table in Streamlit.
        # 'result.style.background_gradient(cmap=cm)' applies background gradient styling to the table using the specified color map.
        # 'set_precision(2)' sets the precision of numerical values in the table to 2 decimal places.
                  
if __name__=='__main__':
    main()
