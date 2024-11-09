# Simple Linear regression
# Simple linear regression is a regression model that estimates the relationship
# between one independent variable and a dependent variable using a straight line.
#The goal is to understand how much sorting time influences delivery time, which could help the company improve efficiency.
# To build a robust model, different transformations (like logarithmic, polynomial, or standard scaling) may be applied to see if they improve accuracy. 
#The performance of each model variation will be evaluated using RMSE (Root Mean Squared Error), which measures prediction accuracy, and the correlation coefficient, which shows the strength of the linear relationship between actual and predicted values. 
#By comparing RMSE and correlation values across models, the company can identify the best-fit model to predict delivery times more reliably based on sorting times.
#Importing Libraries
import pandas as pd #pandas as pd: Imports the pandas library for data handling and manipulation, especially useful for working with tabular data (DataFrames).
import matplotlib.pyplot as plt#matplotlib.pyplot as plt: Imports matplotlib.pyplot to create visualizations such as line charts and scatter plots.
import seaborn as sns#seaborn as sns: Imports seaborn, a visualization library that builds on matplotlib, for statistical data visualization.
import numpy as np#numpy as np: Imports numpy for numerical operations, including functions for linear algebra and random number generation.
from sklearn.linear_model import LinearRegression#LinearRegression: Imports a linear regression model from sklearn.linear_model to build and train a regression model.
from sklearn.metrics import mean_squared_error#mean_squared_error: Imports a metric function to calculate the Mean Squared Error for assessing model performance.
from sklearn.model_selection import train_test_split#train_test_split: Imports a function for splitting the dataset into training and testing subsets, allowing us to evaluate the model’s performance.
#Loading and Inspecting Data
data = pd.read_csv('delivery_time.csv')#data = pd.read_csv('delivery_time.csv'): Loads a CSV file named 'delivery_time.csv' into a DataFrame named data.
# Display the first few rows of the dataset and basic info for initial inspection
data.head()#Displays the first five rows of the dataset, helping to understand its structure and values.
data.info()#Provides a summary of the dataset, including column names, data types, and non-null counts. Useful for checking data completeness and types.
#Plotting Data Distributions and Relationships
# Plot the distribution and relationship between Delivery Time and Sorting Time
plt.figure(figsize=(12, 5))#Initializes a figure with a specified width and height (12x5 inches), setting up space for the upcoming plots.
# Scatter plot for relationship between Delivery Time and Sorting Time
plt.subplot(1, 2, 1)#Divides the figure into a 1-row, 2-column grid, selecting the first cell for the scatter plot.
sns.scatterplot(data=data, x="Sorting Time", y="Delivery Time")#Creates a scatter plot to visualize the relationship between Sorting Time and Delivery Time.
plt.title("Scatter Plot of Sorting Time vs Delivery Time")#Sets the title for the scatter plot
plt.xlabel("Sorting Time")#Label the x-axis respectively.
plt.ylabel("Delivery Time")#Label the y-axis respectively.
#Histogram for Delivery Time
# Distribution of Delivery Time
plt.subplot(1, 2, 2)#Selects the second cell in the grid for the histogram.
sns.histplot(data["Delivery Time"], kde=True)#Creates a histogram of the Delivery Time values to show its distribution.
plt.title("Distribution of Delivery Time")#Adds a kernel density estimate line to the histogram, smoothing the distribution curve
plt.xlabel("Delivery Time")#Set the title and x-axis label for the histogram.
#Displaying Plots
plt.tight_layout()#Adjusts subplots to fit neatly within the figure.
plt.show()#Displays the plots.
#Calculating Correlation Coefficient
# Calculate the correlation coefficient
correlation = data["Sorting Time"].corr(data["Delivery Time"])#Calculates the Pearson correlation coefficient between Sorting Time and Delivery Time, measuring the strength and direction of their linear relationship.
correlation#Stores this correlation value.
#Preparing Data for Model Training
# Separate features and target variable
X = data[["Sorting Time"]]#Selects Sorting Time as the feature (independent variable) and assigns it to X.
y = data["Delivery Time"]#Selects Delivery Time as the target variable (dependent variable) and assigns it to y.
# Split data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#train_test_split(...): Splits X and y into training and testing sets.
#test_size=0.2: Specifies 20% of data for testing and 80% for training.
#random_state=42: Sets a random seed for reproducibility.
# Initialize and fit the linear regression model
model = LinearRegression()#Initializes a linear regression model
model.fit(X_train, y_train)#Trains the model using the training data X_train and y_train.
#Making Predictions on Test Data
y_pred = model.predict(X_test)# Uses the trained model to predict Delivery Time values for the test set X_test.
#Calculating Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#Calculates the mean squared error (MSE) between the actual (y_test) and predicted (y_pred) Delivery Time values.
#Takes the square root of MSE to get the root mean squared error (RMSE), a standard metric indicating the model’s error in predicting Delivery Time.
# Calculate correlation coefficient for predictions
predicted_correlation = np.corrcoef(y_test, y_pred)[0, 1]
#Calculates the Pearson correlation coefficient between the actual and predicted Delivery Time values.
#Extracts the correlation value from the resulting matrix, storing it in predicted_correlation.
rmse, predicted_correlation
#RMSE and predicted_correlation, provide insights into the model’s performance

