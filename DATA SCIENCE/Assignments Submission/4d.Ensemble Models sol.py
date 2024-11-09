#------------------ENSEMBLE TECHNIQUES-------------
#Ensemble techniques combine predictions from multiple models to improve overall performance and achieve more robust, accurate results.
# Importing necessary libraries
import pandas as pd #Used for loading and manipulating the dataset.
from sklearn.model_selection import train_test_split #Splits data into training and testing sets.
from sklearn.tree import DecisionTreeClassifier #Models for classification.
from sklearn.ensemble import RandomForestClassifier #Models for classification.
from sklearn.metrics import classification_report, accuracy_score #Functions to evaluate model performance.
# Load the dataset
data = pd.read_csv('ClothCompany_Data (1).csv')#Reads the CSV file into a DataFrame called data.
# Display the first few rows and basic info
data.head() #Displays the first few rows to examine the structure of the dataset.
data.info()#Provides an overview of column names, data types, and missing values.
# Converting 'Sales' to a binary categorical variable
# Let's set a threshold at the median value of Sales to categorize as 'High' or 'Low'
data['SalesCategory'] = pd.cut(data['Sales'], bins=[0, data['Sales'].median(), data['Sales'].max()], labels=['Low', 'High'])
#pd.cut(): Creates the SalesCategory column by categorizing Sales into two bins:
#Low: Sales values below the median.
#High: Sales values at or above the median.
# Dropping the original 'Sales' column, as it's no longer needed
data = data.drop('Sales', axis=1)#Removes the original Sales column, keeping only the categorized SalesCategory.
# Encoding categorical variables
data = pd.get_dummies(data, columns=['ShelveLoc', 'Urban', 'US'], drop_first=True)
#pd.get_dummies(): Converts categorical features (ShelveLoc, Urban, US) into binary dummy variables, dropping the first category to avoid redundancy.
# Display the updated dataset structure and first few rows
data.info(), data.head()
#Re-checks the structure and first few rows of the modified DataFrame
# Drop any potential missing value from 'SalesCategory' column
data = data.dropna(subset=['SalesCategory'])
#Removes any rows with missing values in SalesCategory.
# Splitting the dataset
X = data.drop('SalesCategory', axis=1)
y = data['SalesCategory']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#X: Features (all columns except SalesCategory).
#y: Target variable (SalesCategory).
#train_test_split(): Divides X and y into training (70%) and test (30%) sets.
# Building and evaluating the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)#DecisionTreeClassifier(): Initializes a Decision Tree classifier.
dt_model.fit(X_train, y_train)#fit(): Trains the Decision Tree model on the training set.
y_pred_dt = dt_model.predict(X_test)#predict(): Predicts SalesCategory on the test set.
# Building and evaluating the Random Forest model
rf_model = RandomForestClassifier(random_state=42)#RandomForestClassifier(): Initializes a Random Forest classifier.
rf_model.fit(X_train, y_train)#fit(): Trains and predicts with the Random Forest model.
y_pred_rf = rf_model.predict(X_test)#predict(): Trains and predicts with the Random Forest model.
# Metrics for Decision Tree
dt_accuracy = accuracy_score(y_test, y_pred_dt)#accuracy_score(): Computes the accuracy of the Decision Tree model on the test set.
dt_report = classification_report(y_test, y_pred_dt)#classification_report(): Provides detailed precision, recall, and F1-score for each category (High and Low) in the test set predictions.
# Metrics for Random Forest
rf_accuracy = accuracy_score(y_test, y_pred_rf)#accuracy_score() : Similar evaluation for the Random Forest model.
rf_report = classification_report(y_test, y_pred_rf)#classification_report(): Similar evaluation for the Random Forest model.
#Displays the accuracy and classification report for both models.
dt_accuracy, dt_report, rf_accuracy, rf_report
#This code completes the process from data loading to model evaluation, providing insights into the classification performance of the models based on different features in the dataset.
