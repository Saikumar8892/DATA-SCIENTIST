# Importing necessary libraries
import pandas as pd  # For data manipulation and analysis
import streamlit as st  # For building interactive web applications
from sqlalchemy import create_engine  # For connecting to databases
from urllib.parse import quote
import pickle, joblib  # For loading the machine learning model and preprocessing pipelines
# Importing the Image class from the Python Imaging Library (PIL)
from PIL import Image

# Loading the machine learning model from the 'knn.pkl' file using pickle
model = pickle.load(open('knn.pkl', 'rb'))

# Loading the preprocessing pipelines from the 'processed1' and 'processed2' files using joblib
ct1 = joblib.load('processed1')


# Function to make predictions using the loaded model
def predict(data, user, pw, db):
    # Create an engine to connect to MySQL database
    engine = create_engine(f"mysql+pymysql://{user}:{quote(pw)}@localhost/{db}")

    # Preprocess the input data
    data.drop(['id'], axis=1, inplace=True)  # Excluding 'id' column
    newprocessed1 = pd.DataFrame(ct1.transform(data), columns = ct1.get_feature_names_out())

    # Make predictions
    predictions = pd.DataFrame(model.predict(newprocessed1), columns=['diagnosis'])
    
    # Concatenate predictions with original data
    final = pd.concat([predictions, data], axis=1)
    
    # Save predictions to the database
    final.to_sql('cancer_predictions', con=engine, if_exists='replace', chunksize=1000, index=False)

    return final

def main():  
    
    image = Image.open("AiSPRY logo.jpg") # Opening an image file named "AiSPRY logo" using the Image.open() function
    st.sidebar.image(image) # Displaying the image in the sidebar of a Streamlit application using the st.sidebar.image() function

    # Title and sidebar title
    st.title("Breast Cancer Prediction")  # Setting the title for the main page
    st.sidebar.title("Breast Cancer Prediction")  # Setting the title for the sidebar
    
    # File uploader in the sidebar to upload CSV or Excel files
    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'], accept_multiple_files=False, key="fileUploader")
    
    # Checking if a file is uploaded
    if uploadedFile is not None:
        try:
            data = pd.read_csv(uploadedFile)  # Trying to read the uploaded file as CSV
        except:
            try:
                data = pd.read_excel(uploadedFile)  # If reading as CSV fails, trying to read as Excel
            except:      
                data = pd.DataFrame(uploadedFile)  # If both fail, treating the file as a DataFrame
    else:
        st.sidebar.warning("You need to upload a csv or excel file.")  # Warning message if no file is uploaded
    
    # Adding a markdown message to the sidebar
    # st.sidebar.markdown(html_temp, unsafe_allow_html=True)
    
    # Text input fields for user, password, and database in the sidebar
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here", type='password')
    db = st.sidebar.text_input("database", "Type Here")
    
    
    
    # Button to trigger prediction
    if st.button("Predict"):
        result = predict(data, user, pw, db)  # Calling the predict function
        
        import seaborn as sns  # Importing seaborn for visualization
        cm = sns.light_palette("blue", as_cmap=True)  # Creating a color map for styling the result table
        st.table(result.style.background_gradient(cmap=cm))  # Displaying the result table with styled background
                           
if __name__=='__main__':
    main()
