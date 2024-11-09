# Import necessary libraries
# Import necessary libraries
import pandas as pd  # Powerful library for data manipulation (loading, cleaning, analysis)
from sklearn.preprocessing import LabelEncoder  # Handles categorical data (text labels) by converting them to numerical values for machine learning models
from sklearn.metrics import accuracy_score  # Evaluates classification model performance (calculates the proportion of correct predictions)

# Load the dataset from a CSV file
wvs = pd.read_csv(r"wvs.csv")  # Reads the CSV file "wvs.csv" located at "C:/Data/" into a pandas DataFrame named "wvs"


# Display the first few rows of the dataset
wvs.head()

# Exploratory Data Analysis (EDA)
wvs.describe()  # Summary statistics
wvs.columns  # List of columns in the dataset

# Convert categorical variables into binary using LabelEncoder
# Initialize a LabelEncoder for encoding categorical data
lb = LabelEncoder()

# Encode categorical columns in the DataFrame
wvs["poverty"] = lb.fit_transform(wvs["poverty"])  # Encode "poverty" column
wvs["religion"] = lb.fit_transform(wvs["religion"])  # Encode "religion" column
wvs["degree"] = lb.fit_transform(wvs["degree"])  # Encode "degree" column
wvs["country"] = lb.fit_transform(wvs["country"])  # Encode "country" column
wvs["gender"] = lb.fit_transform(wvs["gender"])  # Encode "gender" column

# Alternative vectorized approach (more efficient for multiple columns)
# from sklearn.preprocessing import LabelEncoder
# categorical_cols = ["poverty", "religion", "degree", "country", "gender"]
# le = LabelEncoder()
# wvs[categorical_cols] = le.fit_transform(wvs[categorical_cols])
#pip install mord
# Import the ordinal regression model from the mord library
from mord import LogisticAT

# Fit the ordinal regression model to the data
# Fit a Logistic Regression model to the data
model = LogisticAT(alpha=0).fit(wvs.iloc[:, 1:], wvs.iloc[:, 0])  # Train the model on features (all columns except the first) and target variable (first column)


# Access and display the coefficients of the model, which represent
# the estimated weights for each feature's contribution to the prediction.
# Each coefficient indicates the strength and direction of a feature's
# relationship with the target variable.
print(model.coef_)

# Access and display the classes learned by the model during training.
# These are the possible discrete categories or labels that the model
# can predict for new instances.
print(model.classes_)

# Make predictions using the fitted model
predict = model.predict(wvs.iloc[:, 1:])  # Predictions on the training data

# Calculate the accuracy of the model
# Evaluate model performance using accuracy score
accuracy = accuracy_score(wvs.iloc[:, 0], predict)  # Calculate accuracy by comparing true labels (wvs.iloc[:, 0]) with model predictions (predict)

