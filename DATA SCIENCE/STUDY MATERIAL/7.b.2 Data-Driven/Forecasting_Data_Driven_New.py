'''#CRISP-ML(Q)
Business Problem: CocaCola management is reactive in understanding the sales patterns.
They are missing a chance to proactively take actions to meet their sales targets.

Business Objective: Maximize Revenue
Business Constraints: Minimize the gut feel based decisions

Success Criteria: 
    Business: Increase the revenue by at least 20%
    ML: Achieve an accuracy of at least 85%
    Economic: Achieve an increase in revenue by at least $200K

Data Understanding:
    Quaterly sales data from Q1 '86 until Q2 '96. In total we have 42 quarters of data. 
    We have two columns
    Column 1: Quarter
    Column 2: Sales (Target / Output) '''

# Import libraries commonly used for data analysis
import numpy as np  # Used for numerical computations (potentially used later)
import pandas as pd  # Used for data manipulation (reading and working with DataFrame)
import matplotlib.pyplot as plt  # Used for data visualization (potentially used for plotting later)

# Import functions for time series decomposition (exploratory analysis)
from statsmodels.tsa.seasonal import seasonal_decompose  # Decomposes time series data into components

# Import classes for exponential smoothing models (forecasting)
from statsmodels.tsa.holtwinters import SimpleExpSmoothing  # Implements Simple Exponential Smoothing
from statsmodels.tsa.holtwinters import Holt  # Implements Holt's Exponential Smoothing (with trend)
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # Implements general Holt-Winters Exponential Smoothing (with trend and seasonality)

from sqlalchemy import create_engine  # For interacting with databases
from urllib.parse import quote

# Define database credentials (replace with your actual credentials)
user = 'user1'  # Username for database access
pw = quote('amer@mysql')  # Password for database access
db = 'titan'  # Name of the database to connect to

# Create connection engine for interacting with MySQL database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")  # Construct connection string dynamically

# Read Coca-Cola sales data from an Excel file
df = pd.read_excel(r"D:/New materials/DS/Forecasting/Data-Driven/CocaCola_Sales_Rawdata.xlsx")  # Read data using raw string interpretation into database 
# Save the DataFrame 'df' to a table named 'cocacola' in the database
df.to_sql('cocacola', con=engine, if_exists='replace', chunksize=1000, index=False)

# Define a SQL query to select all data from the 'cocacola' table
sql = 'select * from cocacola'

# Load the data from the database using the defined query
cocacola = pd.read_sql_query(sql, con=engine)

# Plot the 'Sales' column of the 'cocacola' DataFrame (likely a time series plot)
cocacola.Sales.plot()

# Splitting the data into Train and Test sets

# Separate the last 4 rows as Test data (assuming recent data for testing)
Test = cocacola.tail(4)

# Select all rows except the last 4 as Train data
Train = cocacola.iloc[:-4]  # Alternative: Train = cocacola.head(len(cocacola)-4)

# Function to calculate Mean Absolute Percentage Error (MAPE)



# Creating a function to calculate the MAPE value for test data 
# Define a function to calculate Mean Absolute Percentage Error (MAPE)
def MAPE(pred, actual):
  """
  This function calculates the Mean Absolute Percentage Error (MAPE) between predicted and actual values.

  Args:
      pred (pandas.Series or list): The predicted values.
      actual (pandas.Series or list): The actual values.

  Returns:
      float: The MAPE value.
  """

  # Calculate the absolute percentage error for each data point
  temp = np.abs((pred - actual) / actual) * 100

  # Return the mean of the absolute percentage errors
  return np.mean(temp)

# Moving Average for Time Series Forecasting

# Calculate a 4-period moving average for the "Sales" column
mv_pred = cocacola["Sales"].rolling(window=4).mean()

# Get the moving average predictions for the test period (last 4 rows)
mv_test_pred = mv_pred.tail(4)

# Calculate MAPE to evaluate moving average forecast accuracy on test data
mape_value = MAPE(mv_test_pred, Test.Sales)
print(f"MAPE for Moving Average Forecast: {mape_value:.2f}%")  # Print MAPE with 2 decimal places

# Plot Sales Data with Moving Averages

# Plot the actual sales data
cocacola.Sales.plot(label="Actual Sales")

# Loop to plot moving averages with different window sizes
for window_size in range(2, 9, 2):
  # Calculate moving average for current window size
  mv_avg = cocacola["Sales"].rolling(window=window_size).mean()
  # Plot the moving average with a label indicating the window size
  mv_avg.plot(label=f"{window_size}-period Moving Average")

# Add legend and title to the plot
# Display the legend at location 3, which is the lower left corner of the plot
plt.legend(loc=3)

# Set the title of the plot to "Coca-Cola Sales Data with Moving Averages"
plt.title("Coca-Cola Sales Data with Moving Averages")

# Show the plot
plt.show()


# ACF and PACF plot on Original data sets 
# Import the plot_acf and plot_pacf functions from the statsmodels.graphics.tsaplots module
import statsmodels.graphics.tsaplots as tsa_plots

# Plot the autocorrelation function (ACF) of the 'Sales' column of the 'cocacola' DataFrame with a lag of 4
tsa_plots.plot_acf(cocacola.Sales, lags=4)
# ACF plot visualizes correlation between time series and its lagged values at different time points.
# The x-axis represents lag values, and the y-axis represents correlation coefficients.
# It helps identify patterns of correlation, indicating potential seasonality and trend.

# Plot the partial autocorrelation function (PACF) of the 'Sales' column of the 'cocacola' DataFrame with a lag of 4
tsa_plots.plot_pacf(cocacola.Sales, lags=4)
# PACF plot measures direct relationship between observations at different time lags, excluding the influence of intermediate lags.
# It helps in identifying the order of the autoregressive (AR) process in time series modeling.


#################### Simple Exponential Method ######################################################################################

# Fit a simple exponential smoothing (SES) model to the training data 'Sales' column using SimpleExpSmoothing
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
# The SimpleExpSmoothing function fits an SES model to the training data by assigning exponentially decreasing weights to past observations.

# Generate predictions using the fitted SES model starting from the first index of the test data to the last index
pred_ses = ses_model.predict(start=Test.index[0], end=Test.index[-1])
# The predict method forecasts future values based on the trained SES model, starting from the first index of the test data to the last index.

# Calculate the Mean Absolute Percentage Error (MAPE) between the predicted and actual sales values in the test data
ses = MAPE(pred_ses, Test.Sales) 
# MAPE (Mean Absolute Percentage Error) is calculated to evaluate the accuracy of the SES model's predictions compared to the actual test data.

##################### Holt method ####################################################################################################

# Fit a Holt-Winters (HW) exponential smoothing model to the training data 'Sales' column using Holt
hw_model = Holt(Train["Sales"]).fit()
# The Holt function fits a Holt-Winters exponential smoothing model to the training data, which captures both trend and seasonality.

# Generate predictions using the fitted HW model starting from the first index of the test data to the last index
pred_hw = hw_model.predict(start=Test.index[0], end=Test.index[-1])
# The predict method forecasts future values based on the trained HW model, starting from the first index of the test data to the last index.

# Calculate the Mean Absolute Percentage Error (MAPE) between the predicted and actual sales values in the test data
hw = MAPE(pred_hw, Test.Sales) 
# MAPE (Mean Absolute Percentage Error) is calculated to evaluate the accuracy of the Holt-Winters model's predictions compared to the actual test data.


#################### Holts winter exponential smoothing with additive seasonality and additive trend ################################

# Fit a Holt-Winters Exponential Smoothing (HWES) model with additive seasonality and additive trend to the training data 'Sales' column
hwe_model_add_add = ExponentialSmoothing(Train["Sales"], seasonal="add", trend="add", seasonal_periods=4).fit()
# The ExponentialSmoothing function fits an HWES model to the training data with additive seasonality and trend components.
# Here, seasonal="add" specifies additive seasonality, trend="add" specifies additive trend, and seasonal_periods=4 specifies the seasonal period as 4.

# Generate predictions using the fitted HWES model starting from the first index of the test data to the last index
pred_hwe_add_add = hwe_model_add_add.predict(start=Test.index[0], end=Test.index[-1])
# The predict method forecasts future values based on the trained HWES model, starting from the first index of the test data to the last index.

# Calculate the Mean Absolute Percentage Error (MAPE) between the predicted and actual sales values in the test data
hwe = MAPE(pred_hwe_add_add, Test.Sales)
# MAPE (Mean Absolute Percentage Error) is calculated to evaluate the accuracy of the HWES model's predictions compared to the actual test data.

#################### Holts winter exponential smoothing with multiplicative seasonality and additive trend ############################

# Fit a Holt-Winters Exponential Smoothing (HWES) model with multiplicative seasonality and additive trend to the training data 'Sales' column
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"], seasonal="mul", trend="add", seasonal_periods=4).fit()
# The ExponentialSmoothing function fits an HWES model to the training data with multiplicative seasonality and additive trend components.
# Here, seasonal="mul" specifies multiplicative seasonality, trend="add" specifies additive trend, and seasonal_periods=4 specifies the seasonal period as 4.

# Generate predictions using the fitted HWES model starting from the first index of the test data to the last index
pred_hwe_mul_add = hwe_model_mul_add.predict(start=Test.index[0], end=Test.index[-1])
# The predict method forecasts future values based on the trained HWES model, starting from the first index of the test data to the last index.

# Calculate the Mean Absolute Percentage Error (MAPE) between the predicted and actual sales values in the test data
hwe_w = MAPE(pred_hwe_mul_add, Test.Sales)
# MAPE (Mean Absolute Percentage Error) is calculated to evaluate the accuracy of the HWES model's predictions compared to the actual test data.


# Create a pandas Series containing Mean Absolute Percentage Error (MAPE) values for different forecasting methods
di = pd.Series({'Simple Exponential Method': ses,  # MAPE for Simple Exponential Smoothing method
                'Holt method ': hw,  # MAPE for Holt method
                'HW with additive seasonality and additive trend': hwe,  # MAPE for Holt-Winters method with additive seasonality and trend
                'HW with multiplicative seasonality and additive trend': hwe_w})  # MAPE for Holt-Winters method with multiplicative seasonality and additive trend

# Create a pandas DataFrame using the Series 'di' with column name 'mape'
mape = pd.DataFrame(di, columns=['mape'])
# The DataFrame 'mape' contains the calculated MAPE values for different forecasting methods

# Final Model on 100% Data
hwe_model_add_add = ExponentialSmoothing(cocacola["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()

# The models and results instances all have a save and load method, so you don't need to use the pickle module directly.
# to save model
hwe_model_add_add.save("model.pickle")

# Import the os module to access operating system functionalities
import os

# Get the current working directory
os.getcwd()

# Import the OLSResults class from the statsmodels.regression.linear_model module to load the model
from statsmodels.regression.linear_model import OLSResults

# Load the pre-trained OLS model from the pickle file named "model.pickle"
model = OLSResults.load("model.pickle")

# Load the new data, which includes entries for future 4 values, from an Excel file
new_data = pd.read_excel(r"Newdata_CocaCola_Sales.xlsx")

# Generate predictions using the loaded model, starting from the first index of the new data to the last index
newdata_pred = model.predict(start=new_data.index[0], end=new_data.index[-1])

# Create a plot to visualize the actual sales values and the predicted sales values for the new data
fig, ax = plt.subplots()
# Plot the actual sales values from the new data as blue line
ax.plot(new_data.Sales, '-b', label='Actual Value')
# Plot the predicted sales values as red line
ax.plot(newdata_pred, '-r', label='Predicted value')
# Add legend to the plot
ax.legend()
# Display the plot
plt.show()

##################################################