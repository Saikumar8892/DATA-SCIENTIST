# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import statsmodels.graphics.tsaplots as tsa_plots  # For time series analysis plots
from statsmodels.tsa.arima.model import ARIMA  # For ARIMA modeling
from sklearn.metrics import mean_squared_error  # For evaluating model performance
from math import sqrt  # To calculate the square root for RMSE
import matplotlib.pyplot as plt  # For plotting visualizations
from sqlalchemy import create_engine  # For database connection
from urllib.parse import quote
# Database credentials (store securely in production environments)
import numpy as np

user = 'root'
password = quote('Sai@123kumar')
db = 'forecasting'

# Connect to MySQL database 'wall_db' using credentials and create an engine
engine = create_engine(f"mysql+pymysql://{user}:{password}@localhost/{db}")

# Read "Walmart Footfalls Raw.csv" into a pandas DataFrame (replace path if necessary)
df = pd.read_csv(r"Walmart Footfalls Raw.csv")

# dumping data into database
# Write the DataFrame 'df' to a table named 'walmart' in the MySQL database
df.to_sql('walmart', con=engine, if_exists='replace', index=False, chunksize=1000)

# Explanation:
#  - Write the DataFrame 'df' to a table named 'walmart'
#  - Con: The database engine object 'engine' created earlier for connection
#  - if_exists: How to handle existing table (replace in this case)
#  - index: Don't write DataFrame index as a separate column (set to False)
#  - chunksize: Write data in chunks of 1000 rows for potentially large DataFrames

# SQL query to select all data from the 'walmart' table
sql = 'select * from walmart'

# Read data from SQL database using the provided SQL query and connection engine
Walmart = pd.read_sql_query(sql, con=engine)

# Partitioning the data into training and testing sets
Train = Walmart.head(147)  # Selecting the first 147 rows as training data
Test = Walmart.tail(12)  # Selecting the last 12 rows as testing data

# Saving the testing data to a CSV file named 'test_arima.csv'
Test.to_csv('test_arima.csv')

# Importing the 'os' module to handle file operations
import os

# Getting the current working directory path
os.getcwd()

# Reading the CSV file 'test_arima.csv' into a pandas DataFrame with index column as 0th column
df = pd.read_csv('test_arima.csv', index_col=0)
# Exploratory Data Analysis (ACF and PACF)

# Exploratory Data Analysis (ACF and PACF)

# Plot the Autocorrelation Function (ACF) of Footfalls (12 lags)
tsa_plots.plot_acf(Walmart['Footfalls'], lags=12)

#  - This line calculates and plots the Autocorrelation Function (ACF) of the 'Footfalls' column in the 'Walmart' DataFrame.
#  - The 'lags' argument specifies the number of lags to consider (12 in this case).
#  - The ACF helps identify potential seasonality or serial dependence in the time series data.

# Plot the Partial Autocorrelation Function (PACF) of Footfalls (12 lags)
tsa_plots.plot_pacf(Walmart['Footfalls'], lags=12)

#  - This line calculates and plots the Partial Autocorrelation Function (PACF) of the 'Footfalls' column in the 'Walmart' DataFrame.
#  - The 'lags' argument specifies the number of lags to consider (12 in this case).
#  - The PACF helps identify the order of the Autoregressive (AR) model needed for forecasting.

# ARIMA Model Fitting and Forecasting

# Create an ARIMA model with AR=12, MA=6 for Train Footfalls
model = ARIMA(Train['Footfalls'], order=(12, 1, 6))
res = model.fit()
print(res.summary())

#  - This line creates an ARIMA model named 'model1' based on the 'Train' DataFrame's 'Footfalls' column.
#  - The 'order' argument specifies the model parameters: AR=12 (Autoregressive order), I=1 (Integrate Differencing - assumed to be 1 here), MA=6 (Moving Average order).
#  - 'res1' stores the results of fitting the model to the training data.
#  - 'print(res1.summary())' displays a summary of the model fit, including statistics like AIC and BIC.

# Forecast Footfalls for the next 12 months
start_index = len(Train)  # Get the index of the last element in Train
end_index = start_index + 11  # Calculate the end index for 12-month forecast
forecast_test = res.predict(start=start_index, end=end_index)

print(forecast_test)

#  - This section forecasts footfalls for the next 12 months using the fitted model 'res1'.
#  - 'start_index' is calculated as the length of the 'Train' DataFrame to begin forecasting from the last training point.
#  - 'end_index' is set 12 steps ahead of 'start_index' to predict 12 months.
#  - 'forecast_test' stores the predicted footfall values for the next 12 months.
#  - 'print(forecast_test)' displays the predicted footfall values.

# Evaluate Model Performance (RMSE)

# Calculate RMSE between actual Test Footfalls and predicted values
rmse_test = sqrt(mean_squared_error(Test.Footfalls, forecast_test))

#  - This line calculates the Root Mean Squared Error (RMSE) between the actual footfalls in the 'Test' DataFrame and the predicted values in 'forecast_test'.
#  - 'mean_squared_error' from 'sklearn.metrics' computes the mean squared difference between the two arrays.
#  - 'sqrt' (square root) is used to convert the mean squared error to RMSE.

# Print Test RMSE (formatted to 3 decimal places)
print('Test RMSE: %.3f' % rmse_test)

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

# Calculate MAPE between actual footfalls in the 'Test' DataFrame and the predicted values in 'forecast_test'
mape = mean_absolute_percentage_error(Test.Footfalls, forecast_test)

# Calculate Test MAPE and format it to 3 decimal places
formatted_mape = '%.3f' % mape
# Print Test MAPE (formatted to 3 decimal places)
print(f'Test MAPE: {formatted_mape}')

# Convert the formatted MAPE back to a float and subtract from 100
Accuracy = 100 - float(formatted_mape) 

# Print the adjusted MAPE in percentage form
print('Accuracy: %.3f%%' % Accuracy)

# Visualize Forecast vs. Actual Footfalls
# Plot the actual footfalls from the Test DataFrame
plt.plot(Test.Footfalls)

#  - This line creates a line plot of the actual footfalls data from the 'Test' DataFrame.

# Plot the predicted footfalls (forecast_test) in red color
plt.plot(forecast_test, color='red')

#  - This line overlays a line plot of the predicted footfalls ('forecast_test') on the same chart in red color.

# Display the plot
plt.show()



#  - This line displays the generated plot, allowing you to visually compare the actual footfalls (blue line) with the predicted values (red line). This helps assess how well the model's forecasts match the real data.

# Auto-ARIMA - Automatically discover the optimal order for an ARIMA model.
# pip install pmdarima --user
# Import pmdarima library for automated ARIMA model selection
import pmdarima as pm

# Get help on the auto_arima function (uncomment to view documentation)
# help(pm.auto_arima)

# Use auto_arima to find potentially good ARIMA parameters
ar_model = pm.auto_arima(Train.Footfalls,
                         start_p=0, start_q=0,  # Start searching from no autoregressive (p) or moving average (q) terms
                         max_p=12, max_q=12,    # Set maximum values to search for p and q
                         m=12,                    # Inform pmdarima of the data's seasonality (monthly in this case)
                         d=None,                   # Let auto_arima determine the differencing order (d)
                         seasonal=True,           # Consider seasonality in the model selection
                         start_P=0,                # Not used in pm.auto_arima according to documentation
                         trace=True,               # Print progress information during search (optional)
                         error_action='warn',      # Raise a warning on potential errors (optional)
                         stepwise=True)            # Use stepwise search for efficiency

# Best Parameters ARIMA (from auto_arima)
# This section is commented out as we're not using the direct output from auto_arima here

# # The line below would likely contain the identified ARIMA order (e.g., (2, 1, 0)) based on auto_arima results
# # model = ARIMA(Train.Footfalls, order=ar_model.order)  # Uncomment if using auto_arima output

# Manually define an ARIMA model with potential optimal parameters (based on pmdarima suggestion or domain knowledge)
model1 = ARIMA(Train.Footfalls, order=(2, 1, 0))  # AR=2, I=1 (assumed), MA=0
res1 = model1.fit()
print(res.summary())  # Print model fitting summary

# Forecast Footfalls for the next 12 months
start_index = len(Train)  # Get the index of the last element in Train
end_index = start_index + 11  # Calculate the end index for 12-month forecast
forecast_best = res1.predict(start=start_index, end=end_index)

print(forecast_best)  # Print the predicted footfall values for the next 12 months

# Evaluate model performance on test set
# RMSE
rmse_best = sqrt(mean_squared_error(Test.Footfalls, forecast_best))
print('Test RMSE: %.3f' % rmse_best)  # Print RMSE for evaluation


# Visualize forecasts against actual outcomes
plt.plot(Test.Footfalls)  # Plot actual footfalls
plt.plot(forecast_best, color='red')  # Overlay predicted values in red
plt.show()  # Display the plot

# Compare model performance with and without Auto-ARIMA
print('Test RMSE with Auto-ARIMA: %.3f' % rmse_best)
print('Test RMSE without Auto-ARIMA: %.3f' % rmse_test)

# Save the better-performing model
res.save("model.pickle")  # Preserve the model with lower RMSE

# Load the saved model for future use
from statsmodels.regression.linear_model import OLSResults  # Import model loading module
model = OLSResults.load("model.pickle")  # Load the saved model

# Forecast for the next 12 months using the loaded model
start_index = len(Walmart)  # Get the index of the last element in Walmart
end_index = start_index + 11  # Set end index for 12-month forecast
forecast = model.predict(start=start_index, end=end_index)  # Generate forecasts

print(forecast)  # Print the predicted footfall values
