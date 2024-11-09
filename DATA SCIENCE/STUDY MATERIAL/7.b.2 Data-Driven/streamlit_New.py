# Import the pandas library for data manipulation
import pandas as pd  

# Import the Streamlit library for building web applications
import streamlit as st  

# Import the OLSResults class from the statsmodels.regression.linear_model module to load the OLS model
from statsmodels.regression.linear_model import OLSResults  

# Import the matplotlib.pyplot module for creating plots
import matplotlib.pyplot as plt  

# Import the create_engine function from the sqlalchemy module for creating database connection engines
from sqlalchemy import create_engine  
from urllib.parse import quote
# Importing the Image class from the Python Imaging Library (PIL)
from PIL import Image

# Load the pre-trained OLS model from the pickle file named "model.pickle"
model = OLSResults.load("model.pickle")

def main():

    image = Image.open("AiSPRY logo.jpg") # Opening an image file named "AiSPRY logo" using the Image.open() function
    st.sidebar.image(image) # Displaying the image in the sidebar of a Streamlit application using the st.sidebar.image() function
    # Set the title of the main page as "Forecasting"
    st.title("Forecasting")
    
    # Set the title of the sidebar as "Forecasting"
    st.sidebar.title("Forecasting")

    # Define HTML content for a styled header
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Forecasting </h2>
    </div>
    
    """
    # Display the styled header using Markdown
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Add a blank line for spacing
    st.text("")


    # Allow the user to upload a file through the sidebar with a file uploader widget
    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'], accept_multiple_files=False, key="fileUploader")
    
    # Check if a file has been uploaded
    if uploadedFile is not None:
        try:
            # Try to read the uploaded file as a CSV file and store it in the variable 'data'
            data = pd.read_csv(uploadedFile)
        except:
            try:
                # If reading as CSV fails, try to read the uploaded file as an Excel file and store it in 'data'
                data = pd.read_excel(uploadedFile)
            except:
                # If reading as both CSV and Excel fails, create a DataFrame using the uploaded file and store it in 'data'
                data = pd.DataFrame(uploadedFile)
    else:
        # If no file has been uploaded, display a warning message in the sidebar
        st.sidebar.warning("You need to upload a CSV or Excel file.")

    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    # Display a styled header in the sidebar using Markdown
    st.sidebar.markdown(html_temp, unsafe_allow_html=True)
    
    # Create text input fields in the sidebar for user input, with default placeholder text "Type Here"
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here", type = 'password')
    db = st.sidebar.text_input("database", "Type Here")
    
    # Check if the "Predict" button is clicked
    if st.button("Predict"):
        # If clicked, create a SQLAlchemy engine for connecting to the MySQL database
        engine = create_engine(f"mysql+pymysql://{user}:{quote(pw)}@localhost/{db}")

        ###############################################
        # Display forecast header in Streamlit (red text)
        st.subheader(":red[Forecast for New data]", anchor=None)
        
        # Generate predictions using the trained model (assuming 'model' is defined)
        newdata_pred = pd.DataFrame(model.predict(start=data.index[0], end=data.index[-1]))
        
        # Combine actual data and predictions into a single DataFrame
        results = pd.concat([data, newdata_pred], axis=1)
        
        # Save forecast results to database table (optional)
        results.to_sql('forecast_results_dd', con=engine, if_exists='replace', index=False, chunksize=1000)
        
        # Import seaborn for styling (might be used for table formatting)
        # Comment out these lines if seaborn styling is not desired
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap=True)
        
        # Display results table in Streamlit with formatting (optional styling)
        st.table(results.style.background_gradient(cmap=cm))  # Uncomment for background gradient (if seaborn imported)

        # Add space for layout (optional)
        st.text("")
        
        # Display forecast plot header in Streamlit (red text)
        st.subheader(":red[plot forecasts against actual outcomes]", anchor=None)
        
        # Generate forecast plot using Matplotlib
        fig, ax = plt.subplots()
        ax.plot(data.Sales, '-b', label='Actual Value')
        ax.plot(newdata_pred, '-r', label='Predicted Value')
        ax.legend()
        st.pyplot(fig)

        
        # data.to_sql('forecast_pred', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        # #st.dataframe(result) or
        # #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        # import seaborn as sns
        # cm = sns.light_palette("blue", as_cmap=True)
        # st.table(result.style.background_gradient(cmap=cm).set_precision(2))

                           
if __name__=='__main__':
    main()


