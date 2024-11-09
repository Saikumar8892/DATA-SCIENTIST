#Problem Statement:Prepare a classification model using the Naive Bayes algorithm for the salary dataset. Train and test datasets are given separately. Use both for model building. And predict Salary
#Note : Do the Deployment by using Streamlit Framework
# Import necessary libraries for data manipulation, model building, and deployment
import pandas as pd  # pandas is used for data manipulation and analysis.
import streamlit as st  # streamlit is a framework for web app development, useful for deploying ML models as web apps.
import joblib  # joblib is used for saving and loading machine learning models.
# Import necessary libraries for data preprocessing
from sklearn.preprocessing import LabelEncoder  # LabelEncoder is used for encoding categorical labels into numerical values.
from sklearn.naive_bayes import GaussianNB  # GaussianNB is a Naive Bayes algorithm suited for continuous data.
from sklearn.preprocessing import OneHotEncoder  # OneHotEncoder is used to create binary columns for each category of categorical features.
# Load train and test datasets
train_data = pd.read_csv('SalaryData_Train.csv')  # Loads the training dataset as a DataFrame.
test_data = pd.read_csv('SalaryData_Test.csv')  # Loads the testing dataset as a DataFrame.
# Display first few rows of each dataset to understand their structure
train_data.head(), test_data.head()  # Shows the first few rows of both train and test data to inspect their columns and types.
# Define a function to preprocess the data by encoding categorical features
def preprocess_data(data):
    le = LabelEncoder()  # Initialize the LabelEncoder instance.
    
    # Apply label encoding to all categorical columns
    for column in data.select_dtypes(include=['object']).columns:  # Loop through all columns with categorical data.
        data[column] = le.fit_transform(data[column])  # Encode each categorical column.
    return data  # Return the preprocessed DataFrame.
# Apply preprocessing on train and test data
train_data_processed = preprocess_data(train_data.copy())  # Preprocess a copy of the training data.
test_data_processed = preprocess_data(test_data.copy())  # Preprocess a copy of the testing data.
# Separate features and target variable for training
X_train = train_data_processed.drop(columns='Salary')  # Remove the target column from the training data features.
y_train = train_data_processed['Salary']  # Extract the target variable.
# Function to preprocess data with label encoders saved for later use
def preprocess_data(data):
    label_encoders = {}  # Initialize a dictionary to store label encoders for each categorical column.
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()  # Initialize LabelEncoder for each categorical column.
        data[column] = le.fit_transform(data[column])  # Transform the categorical column to numeric.
        label_encoders[column] = le  # Save the encoder for future use.
    return data, label_encoders  # Return the preprocessed data and the dictionary of label encoders.
# Load the model
@st.cache
def load_model():
    return joblib.load('naive_bayes_salary_model.pkl')  # Load a pre-trained Naive Bayes model from a file.
# Application title
st.title("Salary Prediction Model")  # Sets the title of the Streamlit app.
# Upload file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")  # Create a file uploader widget for CSV files.
if uploaded_file:  # If a file is uploaded
    test_data = pd.read_csv(uploaded_file)  # Read the uploaded file as a DataFrame.
    st.write("Uploaded Test Data:", test_data.head())  # Display the first few rows of the uploaded data.
    
    # Preprocess data
    test_data_processed, label_encoders = preprocess_data(test_data)  # Preprocess the uploaded data.
    
    # Load model and make predictions
    model = load_model()  # Load the pre-trained model.
    predictions = model.predict(test_data_processed)  # Use the model to predict the target on preprocessed data.
    
    # Display predictions
    test_data['Predicted_Salary'] = predictions  # Add predictions as a new column.
    st.write("Predicted Salary Data:", test_data)  # Display the DataFrame with predictions.
    
    # Download predictions as CSV
    csv = test_data.to_csv(index=False).encode('utf-8')  # Convert DataFrame to CSV format.
    st.download_button("Download Predictions", data=csv, file_name="salary_predictions.csv", mime="text/csv")  # Create a download button.
else:
    st.write("Please upload a test data CSV file to get predictions.")  # Show message if no file is uploaded.
# Separate the features and target in the training data
X_train = train_data.drop(columns=['Salary'])  # Separate features from the target column.
y_train = train_data['Salary']  # Target variable.

# Ensure all columns are of string or numeric type before encoding
X_train = X_train.astype(str)  # Convert features to string type to avoid encoding issues.
test_data_features = test_data.drop(columns=['Salary'], errors='ignore').astype(str)  # Ensure test data columns are string type.

# Initialize OneHotEncoder with handle_unknown='ignore' to avoid issues with unseen categories
encoder = OneHotEncoder(drop='first', handle_unknown='ignore')  # Initialize OneHotEncoder, dropping the first category to avoid redundancy.

# Fit and transform training data; transform test data
X_train_encoded = encoder.fit_transform(X_train).toarray()  # Fit and transform the training data into binary encoded format.
X_test_encoded = encoder.transform(test_data_features).toarray()  # Transform test data based on the training data encoding.

# Initialize and train the Naive Bayes model
nb_model = GaussianNB()  # Initialize Gaussian Naive Bayes model.
nb_model.fit(X_train_encoded, y_train)  # Train the model on encoded training data.

# Predict on the test data
test_predictions = nb_model.predict(X_test_encoded)  # Generate predictions on encoded test data.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your training data
train_data = pd.read_csv('SalaryData_Train.csv')

# Preprocess the data
def preprocess_data(data):
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data, label_encoders

# Preprocess training data
train_data_processed, label_encoders = preprocess_data(train_data.copy())

# Separate features and target
X_train = train_data_processed.drop(columns='Salary')
y_train = train_data_processed['Salary']

# Initialize and train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(nb_model, 'naive_bayes_salary_model.pkl')
print("Model saved as naive_bayes_salary_model.pkl")
