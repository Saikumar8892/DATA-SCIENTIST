# Import necessary libraries
import pandas as pd  # For data manipulation, especially for loading, cleaning, and structuring data in DataFrames
import numpy as np  # For numerical operations, especially for arrays and mathematical functions
import matplotlib.pyplot as plt  # For data visualization, particularly for plotting graphs

from feature_engine.outliers import Winsorizer  # For outlier handling by applying winsorization (capping extreme values)
from sklearn.compose import ColumnTransformer  # To apply different preprocessing steps to different columns in a dataset
from sklearn.impute import SimpleImputer  # For handling missing values by filling them with specified values
from sklearn.preprocessing import StandardScaler  # For scaling features to have a mean of 0 and a standard deviation of 1

from sklearn.pipeline import Pipeline  # To chain multiple preprocessing steps into a single pipeline
import pickle, joblib  # For saving and loading trained models and preprocessing objects

import statsmodels.api as sm  # For statistical modeling, specifically logistic regression in this case
from sklearn.model_selection import train_test_split  # For splitting data into training and test sets
from sklearn import metrics  # For evaluating the model with various performance metrics
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, classification_report  # For classification metrics

# Load the data
data = pd.read_csv('advertising.csv')  # Reads the dataset from a CSV file into a DataFrame
data.info()  # Displays basic information about the data, such as column names, data types, and any missing values

# Drop irrelevant columns if any, based on the dataset exploration
# Assuming 'Ad Topic Line' and 'City' might not be useful for prediction
data = data.drop(['Ad Topic Line', 'City'], axis=1, errors='ignore')  # Removes specified columns if they exist in the data

# Separate features (X) and target variable (Y)
X = data.drop(['Clicked_on_Ad'], axis=1)  # Selects all columns except the target column 'Clicked_on_Ad' for X (features)
Y = data['Clicked_on_Ad']  # Selects the 'Clicked_on_Ad' column for Y (target variable)

# Identify numeric features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns  # Lists the numeric columns for scaling and processing
categorical_features = X.select_dtypes(include=['object']).columns  # Lists the categorical columns (not used here but helpful for future reference)

# Preprocessing pipelines for numeric data
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),  # Fills any missing values in numeric columns with the mean
    ('winsorize', Winsorizer(capping_method='iqr', tail='both', fold=1.5)),  # Caps extreme values in numeric columns based on the IQR (Interquartile Range)
    ('scale', StandardScaler())  # Scales numeric columns to a standard normal distribution (mean=0, standard deviation=1)
])

# ColumnTransformer to apply transformations only to numeric features
preprocess_pipeline = ColumnTransformer([
    ('numerical', num_pipeline, numeric_features)
])

# Fit the preprocessing pipeline and transform the data
X_processed = preprocess_pipeline.fit_transform(X)  # Applies preprocessing steps to X
X_processed = pd.DataFrame(X_processed, columns=numeric_features)  # Converts the transformed data back into a DataFrame with original column names

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_processed, Y, test_size=0.2, random_state=0, stratify=Y)  
# Splits the processed data into training (80%) and testing (20%) sets, stratified by Y to preserve class balance

# Build and fit the logistic regression model using statsmodels
logit_model = sm.Logit(y_train, x_train).fit()  # Fits a logistic regression model on the training data
print(logit_model.summary())  # Displays model summary, including coefficients, p-values, and other statistics

# Save the model and preprocessing pipeline
joblib.dump(preprocess_pipeline, 'preprocess_pipeline.pkl')  # Saves the preprocessing pipeline for future use
pickle.dump(logit_model, open('logistic_model.pkl', 'wb'))  # Saves the trained logistic model

# Generate predictions on training data
y_train_pred = logit_model.predict(x_train)  # Predicts probabilities on the training data

# Calculate the ROC curve for training predictions
fpr, tpr, thresholds = roc_curve(y_train, y_train_pred)  # Computes the False Positive Rate, True Positive Rate, and thresholds
optimal_idx = np.argmax(tpr - fpr)  # Finds the threshold index that maximizes the difference between TPR and FPR
optimal_threshold = thresholds[optimal_idx]  # Sets the optimal threshold based on this index

# Predict and evaluate on training data
y_train_pred_class = (y_train_pred > optimal_threshold).astype(int)  # Converts probabilities to binary predictions using the threshold
print('Training Accuracy:', accuracy_score(y_train, y_train_pred_class))  # Prints the accuracy on the training set
print(classification_report(y_train, y_train_pred_class))  # Prints detailed metrics: precision, recall, and F1 score

# Predict and evaluate on test data
y_test_pred = logit_model.predict(x_test)  # Predicts probabilities on the test set
y_test_pred_class = (y_test_pred > optimal_threshold).astype(int)  # Converts probabilities to binary predictions using the threshold
print('Test Accuracy:', accuracy_score(y_test, y_test_pred_class))  # Prints the accuracy on the test set
print(classification_report(y_test, y_test_pred_class))  # Prints detailed metrics for the test set

# Plot the ROC Curve
plt.plot(fpr, tpr, label="AUC=" + str(metrics.auc(fpr, tpr)))  # Plots the ROC curve with the Area Under Curve (AUC) value
plt.xlabel('False Positive Rate')  # Labels the X-axis
plt.ylabel('True Positive Rate')  # Labels the Y-axis
plt.legend(loc=4)  # Positions the legend in the bottom-right corner
plt.show()  # Displays the ROC curve plot
