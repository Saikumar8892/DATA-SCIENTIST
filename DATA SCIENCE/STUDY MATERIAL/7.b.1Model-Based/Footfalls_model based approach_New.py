'''# CRISP-ML(Q)
Business Problem: Walmart is not aware of the planning because they are unware of the number of customers who will visit their stores.
Business Objective: Maximize Customer Satisfication
Business Constraints: Minimize the number of customer service agents

Success Criteria: 
    Business: Increase the number of footfalls by at least 20%
    ML: Achieve an accuracy of at least 85%
    Economic: Achieve an increase in revenue by at least $200K

Data Understanding:
    Monthly data from Jan '91 until Mar '04. In total we have 159 months of data. 
    We have two columns
    Column 1: Date
    Column 2: Footfalls (Target / Output) '''

# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # Not used in this specific code block, but potentially useful for numerical operations
import pickle  # Not used in this code block, but might be for model saving/loading
from sqlalchemy import create_engine  # For interacting with databases
from urllib.parse import quote
# Database connection credentials (replace with your actual details)
user = 'root'  # Username for database access
pw = quote('Sai@123kumar')  # Password for database access
db = 'forecasting'  # Database name

# Create engine for database connection
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")  # Connect to MySQL database

# Read CSV data into a DataFrame
df = pd.read_csv(r"Walmart Footfalls Raw.csv")  # Load data from specified CSV path

# Store DataFrame as a table in the database
df.to_sql('walmart', con=engine, if_exists='replace', chunksize=1000, index=False)  # Persist data for later access

# Load data back from the database
sql = 'select * from walmart'  # SQL query to retrieve all data

Walmart = pd.read_sql_query(sql, con=engine)  # Execute query and store results in a DataFrame

print(Walmart)  # Display the DataFrame to visually inspect its contents

# Data Pre-processing

# Feature engineering for trend and seasonality

# Add linear trend feature
Walmart["t"] = np.arange(1, 160)  # Create a column "t" with values from 1 to 159 representing a linear trend

# Add quadratic trend feature
Walmart["t_square"] = Walmart["t"] * Walmart["t"]  # Square "t" to capture potential quadratic trend

# Add log-transformed Footfalls feature (for exponential trends)
Walmart["log_footfalls"] = np.log(Walmart["Footfalls"])  # Logarithm of Footfalls to model exponential trends

# Print column names for reference
print(Walmart.columns)

# Explore the "Month" column (assuming it's a string)
p = Walmart["Month"][0]  # Get the first element from "Month"
p[0:3]  # Access the first 3 characters (assuming month abbreviation)

# Create a new "months" column to store month abbreviations
Walmart['months'] = 0  # Initialize with zeros (will be replaced)

for i in range(159):
    p = Walmart["Month"][i]  # Get the i-th element from "Month"
    Walmart['months'][i] = p[0:3]  # Extract the first 3 characters (assuming month abbreviation)

# Create dummy variables for month categories
month_dummies = pd.DataFrame(pd.get_dummies(Walmart['months']))  # One-hot encode month abbreviations
Walmart1 = pd.concat([Walmart, month_dummies], axis=1)  # Combine original with dummy variables
Walmart1 = Walmart1.drop(columns="months")  # Remove the temporary "months" column

# Print column names of the transformed DataFrame
# Print column names of the transformed DataFrame
print(Walmart1.columns)  # Display column names for reference after feature engineering

# Visualize Footfalls over time
Walmart1.Footfalls.plot()  # Plot the "Footfalls" column to explore its trend

# Split data into training and testing sets
Train = Walmart1.head(147)  # First 147 rows for training
Test = Walmart1.tail(12)  # Last 12 rows for testing


# to change the index value in pandas data frame 
# Test.set_index(np.arange(1, 13))
# lets calculate accracy with another methed
# This code defines a function mean_absolute_percentage_error to calculate the mean absolute percentage error (MAPE) 
# between the actual footfalls (y_true) and the predicted footfalls (y_pred)
def mean_absolute_percentage_error(y_true, y_pred): 
    # Convert y_true and y_pred to numpy arrays for ease of computation
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Calculate absolute percentage error for each prediction
    absolute_percentage_error = np.abs((y_true - y_pred) / y_true)
    # Take the mean of all absolute percentage errors and multiply by 100 to get MAPE
    return np.mean(absolute_percentage_error) * 100


####################### Linear ##########################
# Import formula-based API from statsmodels for linear regression
import statsmodels.formula.api as smf

# Build a linear regression model with 't' as the independent variable and 'Footfalls' as the dependent variable
linear_model = smf.ols('Footfalls ~ t', data=Train).fit()  # Train the model on the training data

# Generate predictions using the trained model on the test data's 't' values
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))  # Make predictions for test data

# Calculate Root Mean Squared Error (RMSE) for model evaluation
rmse_linear = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_linear))**2))  # Calculate RMSE
print("RMSE for linear model:", rmse_linear)  # Print the calculated RMSE
# Calculate MAPE between actual footfalls in the 'Test' DataFrame and the predicted values in 'forecast_test'
mape_linear = mean_absolute_percentage_error(Test['Footfalls'], pred_linear)

# Calculate Test MAPE and format it to 3 decimal places
formatted_mape = '%.3f' % mape_linear
# Print Test MAPE (formatted to 3 decimal places)
print(f'Test MAPE: {formatted_mape}')

##################### Exponential ##############################

# Build an exponential regression model
Exp = smf.ols('log_footfalls ~ t', data=Train).fit()  # Train model on log-transformed footfalls

# Generate predictions using the exponential model
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))  # Make predictions on test data's 't'

# Calculate RMSE after converting predictions back from log scale
rmse_Exp = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(np.exp(pred_Exp)))**2))  # Calculate RMSE
print("RMSE for exponential model:", rmse_Exp)  # Print the calculated RMSE

# calculate MAPE
mape_Exp = mean_absolute_percentage_error(Test['Footfalls'], pred_Exp)

# Calculate Test MAPE and format it to 3 decimal places
formatted_mape = '%.3f' % mape_Exp
# Print Test MAPE (formatted to 3 decimal places)
print(f'Test MAPE: {formatted_mape}')

#################### Quadratic ###############################

# Build a quadratic regression model
Quad = smf.ols('Footfalls ~ t + t_square', data=Train).fit()  # Train model with linear and quadratic terms

# Generate predictions using the quadratic model
pred_Quad = pd.Series(Quad.predict(Test[["t", "t_square"]]))  # Make predictions considering both 't' and 't_square'

# Calculate RMSE after predictions
rmse_Quad = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_Quad))**2))  # Calculate RMSE
print("RMSE for quadratic model:", rmse_Quad)  # Print the calculated RMSE

# calculate MAPE
mape_Quad = mean_absolute_percentage_error(Test['Footfalls'], pred_Quad)

# Calculate Test MAPE and format it to 3 decimal places
formatted_mape = '%.3f' % mape_Quad
# Print Test MAPE (formatted to 3 decimal places)
print(f'Test MAPE: {formatted_mape}')

################### Additive Seasonality ########################

# Build an additive seasonal model with dummy variables for months
add_sea = smf.ols('Footfalls ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data=Train).fit()  # Train model with month dummies

# Generate predictions using the additive seasonal model
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']]))  # Make predictions considering month dummies

# Calculate RMSE after predictions
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_add_sea))**2))  # Calculate RMSE
print("RMSE for additive seasonal model:", rmse_add_sea)  # Print the calculated RMSE

# calculate MAPE
mape_add_sea = mean_absolute_percentage_error(Test['Footfalls'], pred_add_sea)

# Calculate Test MAPE and format it to 3 decimal places
formatted_mape = '%.3f' % mape_add_sea
# Print Test MAPE (formatted to 3 decimal places)
print(f'Test MAPE: {formatted_mape}')
################## Multiplicative Seasonality ##################

# Build a multiplicative seasonal model with dummy variables for months (log scale)
Mul_sea = smf.ols('log_footfalls ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data=Train).fit()  # Train model with month dummies on log-transformed footfalls

# Generate predictions using the multiplicative seasonal model
pred_Mult_sea = pd.Series(Mul_sea.predict(Test[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']]))  # Make predictions considering month dummies

# Calculate RMSE after converting predictions back from log scale (for footfalls)
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(np.exp(pred_Mult_sea)))**2))  # Calculate RMSE
print("RMSE for multiplicative seasonal model:", rmse_Mult_sea)  # Print the calculated RMSE

# calculate MAPE
mape_Mult_sea = mean_absolute_percentage_error(Test['Footfalls'], pred_Mult_sea)

# Calculate Test MAPE and format it to 3 decimal places
formatted_mape = '%.3f' % mape_Mult_sea
# Print Test MAPE (formatted to 3 decimal places)
print(f'Test MAPE: {formatted_mape}')

################## Additive Seasonality Quadratic Trend ############################

# Build a combined model with trend, quadratic terms, and seasonal dummies
add_sea_Quad = smf.ols('Footfalls ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data=Train).fit()  # Train model with trend, quadratic, and month dummies

# Generate predictions using the combined model
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 't', 't_square']]))  # Make predictions considering all features

# Calculate RMSE after predictions
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_add_sea_quad))**2))  # Calculate RMSE
print("RMSE for combined model:", rmse_add_sea_quad)  # Print the calculated RMSE

# calculate MAPE
mape_add_sea_quad = mean_absolute_percentage_error(Test['Footfalls'], pred_add_sea_quad)

# Calculate Test MAPE and format it to 3 decimal places
formatted_mape = '%.3f' % mape_add_sea_quad
# Print Test MAPE (formatted to 3 decimal places)
print(f'Test MAPE: {formatted_mape}')

################## Multiplicative Seasonality Linear Trend  ###########

# Build a combined model with trend and seasonal dummies (log scale)
Mul_sea_linear = smf.ols('log_footfalls ~ t + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data=Train).fit()  # Train model with trend and month dummies on log-transformed footfalls

# Generate predictions using the combined model
pred_Mult_sea_linear = pd.Series(Mul_sea_linear.predict(Test[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 't']]))  # Make predictions considering trend and month dummies

# Calculate RMSE after converting predictions back from log scale (for footfalls)
rmse_Mult_sea_linear = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(np.exp(pred_Mult_sea_linear)))**2))  # Calculate RMSE
print("RMSE for combined multiplicative seasonal model:", rmse_Mult_sea_linear)  # Print the calculated RMSE

# calculate MAPE
mape_Mult_sea_linear = mean_absolute_percentage_error(Test['Footfalls'], pred_Mult_sea_linear)

# Calculate Test MAPE and format it to 3 decimal places
formatted_mape = '%.3f' % mape_Mult_sea_linear
# Print Test MAPE (formatted to 3 decimal places)
print(f'Test MAPE: {formatted_mape}')

################## Testing #######################################

# Summarize performance of different models trained earlier
# (likely 'rmse_linear', 'rmse_Exp', etc.)
data = {"MODEL":pd.Series(["rmse_linear", "rmse_Exp", "rmse_Quad", "rmse_add_sea", "rmse_Mult_sea", "rmse_add_sea_quad", "rmse_Mult_sea_linear"]), 
        "RMSE_Values":pd.Series([rmse_linear, rmse_Exp, rmse_Quad, rmse_add_sea, rmse_Mult_sea, rmse_add_sea_quad, rmse_Mult_sea_linear])}
table_rmse = pd.DataFrame(data)
print("Table summarizing RMSE values of different models:")
print(table_rmse)  # Display model names and their corresponding RMSE

# mape table
data1 = {"MODEL":pd.Series(["mape_linear", "mape_Exp", "rmse_Quad", "mape_add_sea", "mape_Mult_sea", "mape_add_sea_quad", "mape_Mult_sea_linear"]), 
        "RMSE_Values":pd.Series([mape_linear, mape_Exp, rmse_Quad, mape_add_sea, mape_Mult_sea, mape_add_sea_quad, mape_Mult_sea_linear])}
table_mape = pd.DataFrame(data1)
print("Table summarizing RMSE values of different models:")
print(table_mape)  # Display model names and their corresponding RMSE


# Identify the model with the least RMSE or mape for forecasting (likely 'rmse_add_sea_quad')
# Build a new forecasting model using the entire data (Walmart1)
# and features from the best performing model ('rmse_add_sea_quad')
model_full = smf.ols('Footfalls ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data = Walmart1).fit()

# Read new data for forecasting from an Excel file (path might need adjustment)
predict_data = pd.read_excel(r"Predict_new.xlsx")

# Generate forecasts for the new data using the trained model ('model_full')
pred_new = pd.Series(model_full.predict(predict_data))
predict_data["forecasted_Footfalls"] = pd.Series(pred_new)

# 'predict_data' now has a new column 'forecasted_Footfalls' with predictions
print("New data with the 'forecasted_Footfalls' column added:")
print(predict_data)  # Display the data with forecasts


# The models and results have save and load method, so you don't need to use the pickle module directly.
# to save model
# Save the trained forecasting model for future use (can be adjusted to a different filename)
model_full.save("model.pickle")

# Check the current working directory (optional, for reference)
import os
print("Current working directory:", os.getcwd())

# Load the saved forecasting model for making predictions on new data later
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model.pickle")

# RESIDUALS MIGHT HAVE ADDITIONAL INFORMATION!

# Autoregression Model (AR)
# Calculating Residuals from best model applied on full data
# AV - FV
full_res = Walmart1.Footfalls - model.predict(Walmart1)

# ACF plot on residuals
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(full_res, lags = 12)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.

# PACF is a partial auto-correlation function. 
# It finds correlations of Y with lags of the residuals of the time series 
tsa_plots.plot_pacf(full_res, lags = 12)

# Alternative approach for ACF plot is explained in next 2 lines
# from pandas.plotting import autocorrelation_plot
# autocorrelation_ppyplot.show()
                          
# AR Autoregressive model
# Build an AutoRegressive (AR) model with lag 1 using Statsmodels

# Import the AutoReg class from the statsmodels.tsa.ar_model module
from statsmodels.tsa.ar_model import AutoReg

model_ar = AutoReg(full_res, lags = [1])
model_fit = model_ar.fit()

# Print the estimated coefficients of the AR(1) model
print('Coefficients: %s' % model_fit.params)

# Generate predictions using the fitted AR(1) model
pred_res = model_fit.predict(start = len(full_res), end = len(full_res) + len(predict_data) - 1, dynamic = False)

# Convert predictions to a Pandas Series and remove index for easier handling
pred_res.reset_index(drop = True, inplace = True)

# Combine forecasts from the previous model ('pred_new') and AR(1) model ('pred_res')
final_pred = pred_new + pred_res

# Display the final predictions after combining forecasts
final_pred


