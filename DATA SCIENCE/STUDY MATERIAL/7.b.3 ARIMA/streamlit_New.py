# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import streamlit as st  # For creating interactive web applications
# import numpy as np  # (Commented out, likely not used in this immediate context)
from statsmodels.regression.linear_model import OLSResults  # For loading a saved model
import matplotlib.pyplot as plt  # For creating visualizations
from sqlalchemy import create_engine  # For interacting with databases
from urllib.parse import quote
from  PIL import Image
# Load the saved model
model = OLSResults.load("model.pickle")  # Retrieve the trained model from its file

def main():
    image = Image.open("AiSPRY logo.jpg")
    st.sidebar.image(image)

    st.title("Forecasting")
    st.sidebar.title("Forecasting")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Forecasting </h2>
    </div>
    
    """
    # Display HTML content using Streamlit markdown with unsafe_allow_html parameter set to True
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("")  # Display an empty line
    
    # Add a file uploader widget to the sidebar for users to upload CSV or Excel files
    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'], accept_multiple_files=False, key="fileUploader")
    
    # Check if a file has been uploaded
    if uploadedFile is not None:
        try:
            # Attempt to read the uploaded file as a CSV file with index column as 0th column
            data = pd.read_csv(uploadedFile, index_col=0)
        except:
            try:
                # If reading as CSV fails, attempt to read the uploaded file as an Excel file with index column as 0th column
                data = pd.read_excel(uploadedFile, index_col=0)
            except:
                # If reading as Excel also fails, treat the uploaded file as raw data (not tabular) and create a DataFrame
                data = pd.DataFrame(uploadedFile)
    else:
        # If no file is uploaded, display a warning message in the sidebar
        st.sidebar.warning("You need to upload a CSV or Excel file.")
        
        
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    # Display HTML content in the sidebar using Streamlit markdown with unsafe_allow_html parameter set to True
    st.sidebar.markdown(html_temp, unsafe_allow_html=True)
    
    # Create text input fields in the sidebar for user to input their username, password, and database name
    user = st.sidebar.text_input("User", "Type Here")  # Text input field for username with default placeholder text "Type Here"
    pw = st.sidebar.text_input("Password", "Type Here", type = 'password')  # Text input field for password with default placeholder text "Type Here"
    db = st.sidebar.text_input("Database", "Type Here")  # Text input field for database name with default placeholder text "Type Here"

    
   # Check if the "Predict" button is clicked
    if st.button("Predict"):
        # Create a SQLAlchemy engine to connect to a MySQL database using the provided username, password, and database name
        engine = create_engine(f"mysql+pymysql://{user}:{quote(pw)}@localhost/{db}")

        ###############################################
       # Display a subheading for the forecast section
        st.subheader(":red[Forecast for Test data]", anchor=None)  # Red header for visual emphasis
        
        # Generate forecasts using the loaded model
        forecast_test = pd.DataFrame(model.predict(start=data.index[0], end=data.index[-1]))  # Predictions for the test data
        
        # Combine actual data and forecasts into a single DataFrame
        results = pd.concat([data, forecast_test], axis=1)  # Merge actual and predicted values
        
        # Store results in a database for persistence
        results.to_sql('forecast_results', con=engine, if_exists='replace', index=False, chunksize=1000)  # Save to database for later access
        
        # Import seaborn for enhanced visualization
        import seaborn as sns  # Leveraging Seaborn for styling
        
        # Create a color palette for visual aesthetics
        cm = sns.light_palette("blue", as_cmap=True)  # Blue gradient for table styling
        
        # Display a visually appealing table of the results
        st.table(results.style.background_gradient(cmap=cm))  # Present results with gradients and precision

        ###############################################
        # Add a blank line for spacing
        st.text("")  # Visual separation in the Streamlit app
        
        # Display a subheading for the plot section
        st.subheader(":red[Plot forecasts against actual outcomes]", anchor=None)  # Red header for emphasis
        
        # Create a plot with Matplotlib
        fig, ax = plt.subplots()  # Set up a new figure and its axes
        
        # Plot the actual Footfalls data
        ax.plot(data.Footfalls)  # Blue line (default color)
        
        # Overlay the forecast_test data in red
        ax.plot(forecast_test, color='red')  # Visual comparison
        
        # Display the plot in the Streamlit app
        st.pyplot(fig)  # Render the plot within the app

        ###############################################
        # Add a blank line for spacing
        st.text("")  # Visual separation in the Streamlit app
        
        # Display a subheading for the forecast section
        st.subheader(":red[Forecast for the next 12 months]", anchor=None)  # Red header for emphasis
        
        # Generate forecasts for the next 12 months
        forecast = pd.DataFrame(model.predict(start=data.index[-1] + 1, end=data.index[-1] + 12))  # Predictions for future periods
        
        # Display the forecast values in a styled table
        st.table(forecast.style.background_gradient(cmap=cm))  # Present results with gradients and precision

        # data.to_sql('forecast_pred', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        # #st.dataframe(result) or
        # #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        # import seaborn as sns
        # cm = sns.light_palette("blue", as_cmap=True)
        # st.table(result.style.background_gradient(cmap=cm).set_precision(2))

                           
if __name__=='__main__':
    main()


