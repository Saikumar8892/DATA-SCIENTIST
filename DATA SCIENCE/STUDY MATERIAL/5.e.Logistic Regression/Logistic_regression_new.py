import pandas as pd  # Import pandas library for data manipulation
import numpy as np  # Import numpy for numerical computing
import matplotlib.pyplot as plt  # Import matplotlib for data visualization

from feature_engine.outliers import Winsorizer  # Import Winsorizer from feature_engine for outlier handling
from sklearn.compose import ColumnTransformer  # Import ColumnTransformer from scikit-learn for preprocessing
from sklearn.impute import SimpleImputer  # Import SimpleImputer from scikit-learn for imputation
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Import preprocessing utilities from scikit-learn

from sklearn.pipeline import Pipeline  # Import Pipeline from scikit-learn for building preprocessing pipelines
import pickle, joblib  # Import pickle and joblib for saving model and preprocessing objects

# import statsmodels.formula.api as smf
import statsmodels.api as sm  # Import statsmodels for statistical modeling
from sklearn.model_selection import train_test_split  # Import train_test_split from scikit-learn for data splitting

from sklearn import metrics  # Import metrics from scikit-learn for model evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, classification_report  # Import metrics for classification

# SQL Integration
from sqlalchemy import create_engine  # Import create_engine from SQLAlchemy for database operations
from urllib.parse import quote
# Create a SQLAlchemy engine for database connection
# Create a SQLAlchemy engine object to connect to the MySQL database
engine = create_engine(
    "mysql+pymysql://{user}:{pw}@localhost/{db}".format(
        user="root",  # Username for MySQL database
        pw=quote("Sai@123kumar"),    # Password for MySQL database
        db="cars" # Name of the MySQL database
    )
)
# Load the offline data into Database to simulate client conditions
claims = pd.read_csv(r"claimants.csv").convert_dtypes()
claims.info()
claims.to_sql('claims', con = engine, if_exists = 'replace', chunksize = 1000, index= False)


# Load the claim data from the MySQL database
sql = 'SELECT * FROM claims'  # SQL query to retrieve all data from the 'claims' table
claimants = pd.read_sql_query(sql, engine).convert_dtypes()  # Read data using the query and engine, converting data types

# Drop the 'CASENUM' column as it might not be relevant for modeling
c1 = claimants.drop('CASENUM', axis=1)  # Drop the 'CASENUM' column

# Separate features (X) and target variable (Y) for modeling
X = c1[['CLMSEX', 'CLMINSUR', 'SEATBELT', 'CLMAGE', 'LOSS']]  # Select features (predictors)
Y = c1[['ATTORNEY']]  # Select target variable

# Identify numeric features (excluding categorical data)
numeric_features = X.select_dtypes(exclude=['object']).columns  # Get all numeric columns
numeric_features1 = X.select_dtypes(include=['int64']).columns  # Get integer columns
numeric_features2 = X.select_dtypes(include=['float64']).columns  # Get float columns

# Define separate imputation techniques for integers and floats
num_pipeline = Pipeline([('impute', SimpleImputer(strategy = 'most_frequent')),
                         ('scale', StandardScaler())])


num_pipeline2 = Pipeline([('impute', SimpleImputer(strategy = 'mean')),
                           ('winsorize', Winsorizer(capping_method = 'iqr', tail='both', fold=1.5)),
                           ('scale', StandardScaler())])

preprocess_pipeline = ColumnTransformer([('numerical', num_pipeline, numeric_features1), 
                                         ('numerical2', num_pipeline2, numeric_features2 )])

clean = preprocess_pipeline.fit(X)  # Pass the raw data through pipeline
clean # imputation scalaing winsorisation encoding 

# Save the defined pipeline
joblib.dump(clean, 'clean') 

import os 
os.getcwd()

# Transform the original data using the pipeline defined above
X3 = pd.DataFrame(clean.transform(X), columns=numeric_features)

X3.info()
# Ensure target variable is integer for compatibility with statsmodels
Y = Y.astype('int')  # Converts target variable to integer data type

# Build the logistic regression model using statsmodels
logit_model = sm.Logit(Y, X3).fit()  # Fits a logistic regression model with Y as target and X3 as features

# Save the fitted model for later use
pickle.dump(logit_model, open('logistic.pkl', 'wb'))  # Stores the fitted model

# Display model summary statistics
logit_model.summary()  # Prints model coefficients, p-values, and goodness-of-fit metrics

# Generate predictions using the fitted model
pred = logit_model.predict(X3)  # Predicts probabilities for each observation in X3

# Calculate optimal threshold for classification
fpr, tpr, thresholds = roc_curve(Y.ATTORNEY, pred)  # Computes false positive rate (fpr), true positive rate (tpr), and thresholds for ROC curve analysis
optimal_idx = np.argmax(tpr - fpr)  # Finds the index of the threshold with the best balance between tpr and fpr
optimal_threshold = thresholds[optimal_idx]  # Extracts the optimal threshold value

# Initialize a new column for predictions (temporarily filled with zeros)
X3["pred"] = np.zeros(len(X3))  # Creates a new column named "pred" and sets all values to zero

# Assign predicted values based on the threshold
X3.loc[pred > optimal_threshold, "pred"] = 1

# Apply optimal threshold for classification
X3.loc[pred > optimal_threshold, "pred"] = 1

# Evaluate model performance
confusion_matrix(X3.pred, Y.ATTORNEY)  # Creates and prints confusion matrix
print('Test accuracy = ', accuracy_score(X3.pred, Y.ATTORNEY))  # Calculates and prints accuracy score
classification = classification_report(X3["pred"], Y)  # Generates classification report
print(classification)  # Prints classification report (precision, recall, F1-score, etc.)

# Plot ROC Curve to visualize model performance
plt.plot(fpr, tpr, label="AUC="+str(metrics.auc(fpr, tpr)))  # Plots ROC curve with AUC (Area Under Curve)
plt.ylabel('True Positive Rate')  # Labels Y-axis
plt.xlabel('False Positive Rate')  # Labels X-axis
plt.legend(loc=4)  # Positions legend in the top right corner
plt.show()  # Displays the ROC curve plot

# Split data into training and testing sets for further evaluation
x_train, x_test, y_train, y_test = train_test_split(X3.iloc[:, :5], Y, test_size=0.2, random_state=0, stratify=Y)
# Explanation of arguments:
#   - x_train, x_test: Separate data for training and testing the model.
#   - y_train, y_test: Separate target variables for training and testing.
#   - test_size=0.2: Allocates 20% of data for testing.
#   - random_state=0: Sets a seed for random splitting (reproducibility).
#   - stratify=Y: Stratifies the split to maintain class proportions in training and testing sets (important for imbalanced datasets).

# Train the logistic regression model on the training data
logisticmodel = sm.Logit(y_train, x_train).fit()  # Fits the model with training features (x_train) and target variable (y_train)

# Evaluate model performance on the training data
y_pred_train = logisticmodel.predict(x_train)  # Predicts probabilities for training observations

# Create a new column "pred" in the training data for predictions (initially filled with zeros)
y_train["pred"] = np.zeros(len(y_train))  # Initializes a new column named "pred" in y_train

# Apply the optimal threshold for classification on training data
y_train.loc[y_pred_train > optimal_threshold, "pred"] = 1  # Sets "pred" to 1 for training observations exceeding the threshold

# Calculate Area Under the ROC Curve (AUC) for training data
auc = metrics.roc_auc_score(y_train["ATTORNEY"], y_pred_train)  # Calculates AUC using actual labels and predicted probabilities
print("Area under the ROC curve (training data) : %f" % auc)

# Generate classification report for the training data
classification_train = classification_report(y_train["pred"], y_train["ATTORNEY"])  # Creates classification report with precision, recall, F1-score, etc.
print(classification_train)  # Prints the classification report for training data

# Evaluate model performance on the testing data
y_pred_test = logisticmodel.predict(x_test)  # Predicts probabilities for testing observations

# Create a new column "y_pred_test" in the testing data for predictions (initially filled with zeros)
y_test["y_pred_test"] = np.zeros(len(y_test))  # Initializes a new column named "y_pred_test" in y_test

# Apply the optimal threshold for classification on testing data
y_test.loc[y_pred_test > optimal_threshold, "y_pred_test"] = 1  # Sets "y_pred_test" to 1 for testing observations exceeding the threshold


# Classification report for testing data
classification1 = classification_report(y_test["y_pred_test"], y_test["ATTORNEY"])
print(classification1)  # Prints classification report with metrics for testing data

# Load the trained logistic regression model
model1 = pickle.load(open('logistic.pkl', 'rb'))  # Loads the saved model

# Load the preprocessing objects (imputation, winsorization, scaler)
clean = joblib.load('clean')  # Loads the saved clean pipeline

# Load new data for prediction

data = pd.read_excel(r"claims_test.xlsx").convert_dtypes()

# Drop the irrelevant 'CASENUM' column
data = data.drop('CASENUM', axis=1)  # Drops the 'CASENUM' column

# Preprocess the new data using the saved transformers
clean = pd.DataFrame(clean.transform(data), columns=data.columns) # Applies imputation using the saved pipeline

# Make predictions on the preprocessed new data
prediction = model1.predict(clean)  # Predicts probabilities for new data observations

# Create a new column "ATTORNEY" for predictions in the new data
data["ATTORNEY"] = np.zeros(len(prediction))  # Initializes "ATTORNEY" column with zeros
data.loc[prediction > optimal_threshold, "ATTORNEY"] = 1  # Sets "ATTORNEY" to 1 for predictions exceeding the threshold

# Ensure "ATTORNEY" is integer for compatibility
data[['ATTORNEY']] = data[['ATTORNEY']].astype('int64')  # Converts "ATTORNEY" to integer data type

# Print the first few rows of the data with predictions
print(data.head())  # Displays the first few rows of the data with predicted labels
