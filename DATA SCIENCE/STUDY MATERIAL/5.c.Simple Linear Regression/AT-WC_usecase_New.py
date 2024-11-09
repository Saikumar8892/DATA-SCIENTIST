# Simple Linear regression
# Simple linear regression is a regression model that estimates the relationship
# between one independent variable and a dependent variable using a straight line.

# Problem Statement
# The Waist Circumference – Adipose Tissue Relationship:
# Studies have shown that individuals with excess Adipose tissue (AT) in 
# their abdominal region have a higher risk of cardiovascular diseases.
# To assess the health conditions of a patient, doctor must get a report 
# on the patients AT values. Computed Tomography, commonly called the CT Scan
# is the only technique that allows for the precise and reliable measurement 
# of the AT (at any site in the body). 

# The problems with using the CT scan are:
# - Many physicians do not have access to this technology
# - Irradiation of the patient (suppresses the immune system)
# - Expensive
# 
# The Hospital/Organization wants to find an alternative solution for this 
# problem, which can allow doctors to help their patients efficiently.
# 

# CRISP-ML(Q) process model describes six phases:
# - Business and Data Understanding
# - Data Preparation (Data Engineering)
# - Model Building (Machine Learning)
# - Model Evaluation and Tuning
# - Deployment
# - Monitoring and Maintenance

# Objective(s): Minimize the risk for patients or maximize the convenience to doctors in assisting their patients
# Constraints: CT Scan is the only option
# Research: A group of researchers conducted a study with the aim of predicting abdominal AT area using simple anthropometric measurements, i.e., measurements on the human body
 
# Proposed Plan:
# The Waist Circumference – Adipose Tissue data is a part of this study wherein
# the aim is to study how well waist circumference (WC) predicts the AT area
# 
# 
# Benefits:
# Is there a simpler yet reasonably accurate way to predict the AT area? i.e.,
# - Easily available
# - Risk free
# - Inexpensive

# Data Collection
# Data: 
#     AT values from the historical Data
#     Waist Circumference of these patients.
# 
# Collection:
# 1. Evaluate the available Hospital records for relevant data (CT scan of patients)
# 
# 2. Record the Waist Circumference of patients - Primary Data
# 
# - Strategy to Collection Primary Data:
#     Call out the most recent patients (1 year old) with an offer of free 
#     consultation from a senior doctor to attract them to visit hospital.
#     Once the patients visit the hospital, we can record their Waist 
#     Circumference with accuracy.

# Explore the Patients Database (MySQL)

# Connect to the MySQL DB source for Primary data
# Load the datasets into Python dataframe
import pandas as pd
from urllib.parse import quote
# Load the patient AT data
data = pd.read_csv(r"ATpatients.csv")

# Load the patient waist data
data2 = pd.read_csv(r"waist.csv")

# Import the library for connecting to databases
from sqlalchemy import create_engine

# 1. Establish a connection to a MySQL database:
#   - Create an engine object, providing authentication details and database name.
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                     .format(user="root",  # Database username
                             pw=quote("Sai@123kumar"),   # Database password
                             db="slr"))  # Database name

# 2. Write data to separate tables:
#   - Write patient AT data to a table named 'atpatients'.
#      - 'if_exists='replace': Overwrites the table if it already exists.
#      - 'index=False': Prevents writing the DataFrame index as a separate column.
data.to_sql('atpatients', con=engine, if_exists='replace', index=False)

#   - Write patient waist data to a table named 'waist'.
data2.to_sql('waist', con=engine, if_exists='replace', index=False)

# 3. Set primary keys:
#   - Define a primary key in both tables to ensure data integrity and efficient retrieval.
from sqlalchemy import text
with engine.connect() as con:
    con.execute(text('ALTER TABLE `atpatients` ADD PRIMARY KEY (`Patient`);')) # Primary key for 'atpatients' table
with engine.connect() as con:
   con.execute(text('ALTER TABLE `waist` ADD PRIMARY KEY (`Patient`);'))  # Primary key for 'waist' table

# 4. Retrieve selected data:
#   - Construct an SQL query to fetch only necessary features from both tables, joining them on the 'Patient' column.
sql = "SELECT A.Patient, A.AT, A.Sex, A.Age, B.Waist from atpatients as A Inner join waist as B on A.Patient = B.Patient;"

#   - Execute the query and store the results in a pandas DataFrame for further processing.
wcat_full = pd.read_sql_query(sql, engine)


# Display basic information about the DataFrame
# Analyze and prepare the data for regression modeling

# 1. Get basic information about the data (wcat_full)
wcat_full.info()  # Print information like data types, number of non-null values, etc. for each column

# 2. Data Cleaning (Feature Selection):
#    - Create a new DataFrame (wcat) by dropping irrelevant features for your regression analysis.
#    - Dropped features here are 'Patient', 'Sex', and 'Age' based on your problem definition.
wcat = wcat_full.drop(["Patient", "Sex", "Age"], axis=1)

# 3. Analyze the DataFrame after Dropping Features (wcat)
wcat.info()  # Print information about the DataFrame 'wcat' after dropping features

# Note: Depending on your specific analysis, you might need to perform additional data cleaning steps like handling missing values or outliers.

# Import necessary libraries for data manipulation, mathematical calculations, visualization, and modeling
import pandas as pd  # Data Manipulation
import numpy as np   # Mathematical calculations
import matplotlib.pyplot as plt  # Data Visualization
import seaborn as sns  # Data Visualization
import joblib  # Saving and loading model
import pickle  # Saving and loading model
from sklearn.compose import ColumnTransformer  # Column Transformer
from sklearn.pipeline import Pipeline  # Pipeline for modeling
from sklearn.impute import SimpleImputer  # Imputing missing values
from sklearn.pipeline import make_pipeline  # Pipeline for modeling
from feature_engine.outliers import Winsorizer  # Handling outliers
from sklearn.model_selection import train_test_split  # Splitting data into train and test sets
import statsmodels.formula.api as smf  # Statsmodels for statistical modeling
from sklearn.preprocessing import PolynomialFeatures  # Polynomial features for modeling
from sklearn.linear_model import LinearRegression  # Linear regression model

# Load the data from a CSV file
wcat = pd.read_csv(r"wc-at.csv")

# Exploratory Data Analysis (EDA) - Get basic information about the data
print("Summary of the data:")
wcat.describe()  # Provides statistics like mean, standard deviation, etc. for each column

# View the first 10 rows of the data
print("First 10 rows of the data:")
wcat.head(10)

# Sort data by waist circumference (ascending order)
wcat.sort_values('Waist', ascending=True, inplace=True)  # Sorts in-place
# Reset the index after sorting (optional, but keeps indexing clean)
wcat.reset_index(inplace=True, drop=True)

# View the first 10 rows after sorting
print("First 10 rows after sorting by waist:")
wcat.head(10)

# Split data into target variable (AT) and predictor variable (Waist)
X = pd.DataFrame(wcat['Waist'])  # Create DataFrame for predictor
Y = pd.DataFrame(wcat['AT'])    # Create DataFrame for target variable

# Select numeric features for data preprocessing
numeric_features = ['Waist']
# Exploratory Data Analysis (EDA) - Visualize outliers with boxplots
print("Boxplots to visualize outliers:")
wcat.plot(kind='box', subplots=True, sharey=False, figsize=(15, 8))
plt.subplots_adjust(wspace=0.75)  # Adjust spacing between subplots
plt.show()

# Winsorize outliers (optional, commented out here)
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['Waist'])

# Define pipelines for data preprocessing (commented out for now)
# Impute missing values with mean
# Winsorize outliers

num_pipeline = Pipeline([('impute', SimpleImputer(strategy = 'mean')),
                         ('winsorize', Winsorizer(capping_method = 'iqr', tail='both', fold=1.5))])

# Define ColumnTransformer for preprocessing (commented out for now)
preprocessor = ColumnTransformer([('numerical', num_pipeline, numeric_features)])  # Preprocess numerical features

# Fit preprocessing pipelines to data (commented out for now)
Clean_data = preprocessor.fit(X)

wcat['Waist'] = pd.DataFrame(Clean_data.transform(X))  # Transform waist column with imputation

# Save the data preprocessing pipelines (commented out for now)
joblib.dump(Clean_data, 'preprocessed')



# Visualize outliers after preprocessing (assuming preprocessing is done)
print("Boxplots to visualize outliers after preprocessing:")
wcat.plot(kind='box', subplots=True, sharey=False, figsize=(15, 8))
plt.subplots_adjust(wspace=0.75)  # Adjust spacing between subplots
plt.show()

# Graphical Representation of the Data
# 1. Bar Graph of Target Variable (AT)
plt.figure(figsize=(10, 6))  # Set figure size for better visualization
plt.bar(height=wcat.AT, x=np.arange(1, 110, 1))  # Create bar graph with index as x-axis
plt.xlabel('Index')  # Label the x-axis
plt.ylabel('AT Value')  # Label the y-axis
plt.title('Bar Graph of AT Values')  # Add a title for clarity
plt.show()  # Display the bar graph

# 2. Histogram of Target Variable (AT)
plt.figure(figsize=(10, 6))  # Set figure size for better visualization
plt.hist(wcat.AT)  # Create a histogram of AT values
plt.xlabel('AT Value')  # Label the x-axis
plt.ylabel('Frequency')  # Label the y-axis
plt.title('Histogram of AT Values')  # Add a title for clarity
plt.show()  # Display the histogram

# 3. Bar Graph of Predictor Variable (Waist)
plt.figure(figsize=(10, 6))  # Set figure size for better visualization
plt.bar(height=wcat.Waist, x=np.arange(1, 110, 1))  # Create bar graph with index as x-axis
plt.xlabel('Index')  # Label the x-axis
plt.ylabel('Waist Circumference')  # Label the y-axis
plt.title('Bar Graph of Waist Circumference')  # Add a title for clarity
plt.show()  # Display the bar graph

# 4. Histogram of Predictor Variable (Waist)
plt.figure(figsize=(10, 6))  # Set figure size for better visualization
plt.hist(wcat.Waist)  # Create a histogram of waist circumference values
plt.xlabel('Waist Circumference')  # Label the x-axis
plt.ylabel('Frequency')  # Label the y-axis
plt.title('Histogram of Waist Circumference')  # Add a title for clarity
plt.show()  # Display the histogram


# The above are manual approach to perform Exploratory Data Analysis (EDA). The alternate approach is to Automate the EDA process using Python libraries.
# 
# Auto EDA libraries:
# - Sweetviz
# - dtale
# - pandas profiling
# - autoviz

# 
# # **Automating EDA with Sweetviz:**
# 

# Using sweetviz to automate EDA is pretty simple and straight forward. 3 simple steps will provide a detailed report in html page.
# 
# step 1. Install sweetviz package using pip.
# - !pip install sweetviz
# 
# step2. import sweetviz package and call analyze function on the dataframe.
# 
# step3. Display the report on a html page created in the working directory with show_html function.

# Import Sweetviz library for automated EDA (optional)
import sweetviz as sv

# Analyze the data using Sweetviz (optional)
# This can generate a comprehensive HTML report summarizing the data.
if __name__ == "__main__":  # Run this block only if the script is executed directly
    report = sv.analyze(wcat)
    report.show_html('EDAreport.html')  # Save the report as an HTML file

# Bivariate Analysis - Explore the relationship between Waist and AT
# 1. Scatter Plot
plt.scatter(x=wcat['Waist'], y=wcat['AT'])
plt.xlabel('Waist Circumference')  # Label the x-axis
plt.ylabel('AT Value')  # Label the y-axis
plt.title('Scatter Plot of Waist vs AT')  # Add a title for clarity
plt.show()  # Display the scatter plot

# 2. Correlation Coefficient - Measure the strength of the linear relationship
correlation = np.corrcoef(wcat.Waist, wcat.AT)[0, 1]
print("Correlation Coefficient between Waist and AT:", correlation)
# Values closer to 1 or -1 indicate stronger linear relationships.

# 3. Covariance - Measure the direction and strength of the joint variability
covariance = np.cov(wcat.Waist, wcat.AT)[0, 1]
print("Covariance between Waist and AT:", covariance)
# A positive covariance suggests both variables tend to move in the same direction,
# while a negative covariance suggests they move in opposite directions.

# 4. Heatmap to visualize correlations between all variables
dataplot = sns.heatmap(wcat.corr(), annot=True, cmap="YlGnBu")  # Create heatmap
plt.title('Correlation Heatmap')  # Add a title for clarity
plt.show()  # Display the heatmap

# Linear Regression Modeling
# 1. Model definition (using statsmodels)
model = smf.ols('AT ~ Waist', data=wcat).fit()  # Fit a simple linear regression model

# 2. Model Summary
print("Linear Regression Model Summary:")
print(model.summary())  # Display various statistics about the model

# 3. Predictions
pred1 = model.predict(pd.DataFrame(wcat['Waist']))  # Predict AT values based on waist

# 4. Visualization of the regression line
plt.scatter(wcat.Waist, wcat.AT)  # Plot the original data points
plt.plot(wcat.Waist, pred1, "r")  # Plot the regression line in red
plt.xlabel('Waist Circumference')  # Label the x-axis
plt.ylabel('AT Value')  # Label the y-axis
plt.title('Linear Regression Line (AT ~ Waist)')  # Add a title for clarity
plt.legend(['Observed data', 'Predicted line'])  # Add a legend
plt.show()  # Display the plot

# Error Calculation for the Base Model
# 1. Calculate residuals (errors) for each data point
res1 = wcat.AT - pred1  # Actual AT values - Predicted AT values
print("Mean of residuals (should be close to zero for good fit):", np.mean(res1))

# 2. Calculate model evaluation metrics
res_sqr1 = res1 * res1  # Square the residuals
mse1 = np.mean(res_sqr1)  # Mean Squared Error
rmse1 = np.sqrt(mse1)  # Root Mean Squared Error
print("Root Mean Squared Error (RMSE) for the base model:", rmse1)

# Model Tuning with Transformations
# 1. Log Transformation of Predictor Variable
# - Visualize relationship after log transformation
# Scatter Plot with Log-Transformed Waist
plt.scatter(x=np.log(wcat['Waist']), y=wcat['AT'], color='brown')  # Create a scatter plot with log-transformed Waist and AT
plt.xlabel('Log(Waist Circumference)')  # Set the x-axis label
plt.ylabel('AT Value')  # Set the y-axis label
plt.title('Scatter Plot with Log-Transformed Waist')  # Set the title of the plot
plt.show()  # Display the plot

# Calculate correlation for transformed data
print("Correlation after log transformation of Waist:", np.corrcoef(np.log(wcat.Waist), wcat.AT)[0, 1])  # Print the correlation coefficient

# Fit Linear Regression with Log-Transformed Predictor
model2 = smf.ols('AT ~ np.log(Waist)', data=wcat).fit()  # Fit a linear regression model with log-transformed Waist
print("Model Summary for log-transformed model:")  # Print model summary message
print(model2.summary())  # Display the model summary

# Predictions and Visualization for Log-Transformed Model
pred2 = model2.predict(pd.DataFrame(wcat['Waist']))  # Generate predictions based on the model
plt.scatter(np.log(wcat.Waist), wcat.AT)  # Scatter plot of log-transformed Waist vs. AT
plt.plot(np.log(wcat.Waist), pred2, "r")  # Plot the regression line
plt.xlabel('Log(Waist Circumference)')  # Set the x-axis label
plt.ylabel('AT Value')  # Set the y-axis label
plt.title('Regression Line with Log-Transformed Waist')  # Set the title of the plot
plt.legend(['Observed data', 'Predicted line'])  # Add legend to the plot
plt.show()  # Display the plot

# Error Calculation for Log-Transformed Model
res2 = wcat.AT - pred2  # Calculate residuals
res_sqr2 = res2 * res2  # Square the residuals
mse2 = np.mean(res_sqr2)  # Calculate mean squared error
rmse2 = np.sqrt(mse2)  # Calculate root mean squared error
print("RMSE for log-transformed model:", rmse2)  # Print the RMSE

# Scatter Plot with Exponential-Transformed AT
plt.scatter(x=wcat['Waist'], y=np.log(wcat['AT']), color='orange')  # Create a scatter plot with Waist and log-transformed AT
plt.xlabel('Waist Circumference')  # Set the x-axis label
plt.ylabel('Log(AT Value)')  # Set the y-axis label
plt.title('Scatter Plot with Exponential-Transformed AT')  # Set the title of the plot
plt.show()  # Display the plot

# Calculate correlation for transformed data
print("Correlation after exponential transformation of AT:", np.corrcoef(wcat.Waist, np.log(wcat.AT))[0, 1])  # Print the correlation coefficient

# Fit Linear Regression with Exponential-Transformed Response
model3 = smf.ols('np.log(AT) ~ Waist', data=wcat).fit()  # Fit a linear regression model with Waist and log-transformed AT
print("Model Summary for exponential-transformed model:")  # Print model summary message
print(model3.summary())  # Display the model summary

# Predictions and Visualization for Exponential-Transformed Model
pred3 = model3.predict(pd.DataFrame(wcat['Waist']))  # Generate predictions based on the model
plt.scatter(wcat.Waist, np.log(wcat.AT))  # Scatter plot of Waist vs. log-transformed AT
plt.plot(wcat.Waist, pred3, "r")  # Plot the regression line
plt.xlabel('Waist Circumference')  # Set the x-axis label
plt.ylabel('Log(AT Value)')  # Set the y-axis label
plt.title('Regression Line with Exponential-Transformed AT')  # Set the title of the plot
plt.legend(['Observed data', 'Predicted line'])  # Add legend to the plot
plt.show()  # Display the plot

# Error Calculation for Exponential-Transformed Model
pred3_at = np.exp(pred3)  # Convert predicted log values back to AT values
res3 = wcat.AT - pred3_at  # Calculate residuals
res_sqr3 = res3 * res3  # Square the residuals
mse3 = np.mean(res_sqr3)  # Calculate mean squared error
rmse3 = np.sqrt(mse3)  # Calculate root mean squared error
print("RMSE for exponential-transformed model:", rmse3)  # Print the RMSE

# Comparing Model Performance
print("The base model has a RMSE of:", rmse1)  # Print RMSE of the base model
print("The log-transformed model has a RMSE of:", rmse2)  # Print RMSE of the log-transformed model
print("The exponential-transformed model has a RMSE of:", rmse3)  # Print RMSE of the exponential-transformed model

# Conclusion (based on RMSE comparison, choose the best model)
# Based on the RMSE values, you can choose the model that performs best. 
# A lower RMSE indicates a better fit for the data.

# Note: This code demonstrates trying different transformations. 
# You can explore other transformations or techniques 
# to improve the model's performance.


# Fit a polynomial regression model
model4 = smf.ols('np.log(AT) ~ Waist + I(Waist*Waist)', data=wcat).fit()  # Fit the model using log-transformed AT and polynomial terms of Waist
model4.summary()  # Display the summary of the model

# Make predictions using the polynomial model
pred4 = model4.predict(pd.DataFrame(wcat))  # Generate predictions based on the model

# Visualize the regression lines for the polynomial and linear models
plt.scatter(X['Waist'], np.log(Y['AT']))  # Scatter plot of Waist vs. log-transformed AT
plt.plot(X['Waist'], pred4, color='red')  # Plot the polynomial regression line
plt.plot(X['Waist'], pred3, color='green', label='linear')  # Plot the linear regression line
plt.legend(['Transformed Data', 'Polynomial Regression Line', 'Linear Regression Line'])  # Add legend to the plot
plt.show()  # Display the plot

# Calculate errors for the polynomial model
pred4_at = np.exp(pred4)  # Transform predictions back to original scale
res4 = wcat.AT - pred4_at  # Calculate residuals
res_sqr4 = res4 * res4  # Square the residuals
mse4 = np.mean(res_sqr4)  # Calculate mean squared error
rmse4 = np.sqrt(mse4)  # Calculate root mean squared error
rmse4  # Display the RMSE

# Create a table to compare RMSE of different models
data = {"MODEL": pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), 
        "RMSE": pd.Series([rmse1, rmse2, rmse3, rmse4])}  # Create a dictionary with model names and RMSE values
table_rmse = pd.DataFrame(data)  # Create a DataFrame to display the RMSE values

table_rmse  # Display the RMSE comparison table

# Split the data into training and testing sets
train, test = train_test_split(wcat, test_size=0.2, random_state=0)  # Split the data into training and testing sets

# Visualize the data points in the training and testing sets
plt.scatter(train.Waist, np.log(train.AT))  # Scatter plot of Waist vs. log-transformed AT in the training set
plt.scatter(test.Waist, np.log(test.AT))  # Scatter plot of Waist vs. log-transformed AT in the testing set

# Fit the final model on the training data
finalmodel = smf.ols('np.log(AT) ~ Waist + I(Waist*Waist)', data=train).fit()  # Fit the final model using the training data

# Make predictions on the test data using the final model
test_pred = finalmodel.predict(test)  # Generate predictions on the test set
pred_test_AT = np.exp(test_pred)  # Transform predictions back to original scale

# Model Evaluation on Test Data

# Calculate the error (residuals) between actual AT values and predicted AT values on the test set
test_res = test.AT - pred_test_AT  # Residuals (errors) for test data

# Square the residuals to get squared errors
test_sqrs = test_res * test_res

# Calculate the Mean Squared Error (MSE) for the test set
test_mse = np.mean(test_sqrs)  # Average of squared errors

# Calculate the Root Mean Squared Error (RMSE) for the test set
# RMSE is a common metric to evaluate the magnitude of the errors
test_rmse = np.sqrt(test_mse)
print("Test RMSE:", test_rmse)  # Print the RMSE on the test set

# Predictions on the Train Data (optional)

# Make predictions on the training data using the final model
train_pred = finalmodel.predict(pd.DataFrame(train['Waist']))

# Convert the predicted log values back to AT scale for the training data
pred_train_AT = np.exp(train_pred)

# Model Evaluation on Train Data (optional)

# Calculate the error (residuals) between actual AT values and predicted AT values on the training set
train_res = train.AT - pred_train_AT  # Residuals (errors) for training data

# Square the residuals to get squared errors
train_sqrs = train_res * train_res

# Calculate the Mean Squared Error (MSE) for the training set
train_mse = np.mean(train_sqrs)  # Average of squared errors

# Calculate the Root Mean Squared Error (RMSE) for the training set
train_rmse = np.sqrt(train_mse)
print("Train RMSE:", train_rmse)  # Print the RMSE on the training set

# Saving the Best Model (Polynomial with Degree 2) for Pipelining

# Create a pipeline that combines polynomial feature creation (degree=2 for quadratic terms)
# with linear regression
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# Fit the pipeline on the features ('Waist') and target ('AT') from the wcat DataFrame
poly_model.fit(wcat[['Waist']], wcat[['AT']])

# Serialize the trained pipeline (poly_model) for later use
pickle.dump(poly_model, open('poly_model.pkl', 'wb'))
print("Saved the polynomial regression model to poly_model.pkl")

# Loading and Using the Saved Model (assuming imputation and winsorization were done previously)

# Load the previously saved imputation model (replace 'meanimpute' with the actual filename)
impute = joblib.load('meanimpute')

# Load the previously saved winsorization model (replace 'winzor' with the actual filename)
winsor = joblib.load('winzor')

# Load the serialized polynomial regression model pipeline
poly_model = pickle.load(open('poly_model.pkl', 'rb'))

# Load the test dataset (replace the file path with your actual location)
wcat_test = pd.read_csv(r"wc-at_test.csv")

# Apply the same preprocessing steps (imputation and winsorization) used during training to the test data
clean1 = pd.DataFrame(impute.transform(wcat_test), columns=wcat_test.select_dtypes(exclude=['object']).columns)
clean2 = pd.DataFrame(winsor.transform(clean1), columns=clean1.columns)

# Make predictions on the preprocessed test data using the loaded polynomial regression model
prediction = pd.DataFrame(poly_model.predict(clean2), columns=['Pred_AT'])

# Concatenate the predictions with the original test data for easier analysis
final = pd.concat([prediction, wcat_test], axis=1)
print(final.head())  # Display the first few rows of the final DataFrame


