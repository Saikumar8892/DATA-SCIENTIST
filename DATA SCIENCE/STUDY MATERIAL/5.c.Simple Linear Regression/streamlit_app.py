import pandas as pd  # Importing pandas library for data manipulation
import streamlit as st  # Importing Streamlit library for building web applications
from PIL import Image # Importing the Image class from the Python Imaging Library (PIL)
from sqlalchemy import create_engine  # Importing create_engine from SQLAlchemy to interact with databases
import pickle, joblib  # Importing pickle and joblib for loading models
from urllib.parse import quote
# Load preprocessing models
Clean = joblib.load('preprocessed')

poly_model = pickle.load(open('poly_model.pkl', 'rb'))

# Define function to predict Average Temperature (AT)
def predict_AT(data, user, pw, db):
    # Creating database engine using provided credentials
    engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))
    
    # Applying preprocessing
    clean1 = pd.DataFrame(Clean.transform(data), columns=data.select_dtypes(exclude=['object']).columns)
    
    
    
    # Making predictions using the trained polynomial regression model
    prediction = pd.DataFrame(poly_model.predict(clean1), columns=['Pred_AT'])
    
    # Combining predictions with original data
    final = pd.concat([prediction, data], axis=1)
    
    # Writing predictions to MySQL database table 'mpg_predictions'
    final.to_sql('mpg_predictions', con=engine, if_exists='replace', chunksize=1000, index=False)

    return final

# Define main function to build Streamlit app
def main():
    image = Image.open("AiSPRY logo.jpg") # Opening an image file named "AiSPRY logo" using the Image.open() function
    st.sidebar.image(image) # Displaying the image in the sidebar of a Streamlit application using the st.sidebar.image() function
    # Setting up title for Streamlit app
    st.title("AT prediction")
    st.sidebar.title("Fuel Efficiency prediction")
    
    # Adding HTML template for app header
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">AT prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Sidebar for uploading file
    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'], accept_multiple_files=False, key="fileUploader")
    
    # Handling file upload
    if uploadedFile is not None:
        try:
            data = pd.read_csv(uploadedFile)  # Attempt to read uploaded file as CSV
        except:
            try:
                data = pd.read_excel(uploadedFile)  # Attempt to read uploaded file as Excel
            except:
                data = pd.DataFrame()  # If file format is not supported, create empty DataFrame
        
    else:
        st.sidebar.warning("You need to upload a CSV or an Excel file.")
    
    # Sidebar for database credentials
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credentials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html=True)
            
    user = st.sidebar.text_input("user", "root")  # Text input for database username
    pw = st.sidebar.text_input("password", "Sai@123kumar")  # Text input for database password
    db = st.sidebar.text_input("database", "knn")  # Text input for database name
    
    result = ""  # Initialize result variable
    
    # Button to trigger prediction
    if st.button("Predict"):
        result = predict_AT(data, user, pw, db)  # Call predict_AT function to make predictions
        
        # Display predictions in a table format with color gradient
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap=True)
        st.table(result.style.background_gradient(cmap=cm))

# Run main function if script is executed directly
if __name__ == '__main__':
    main()
