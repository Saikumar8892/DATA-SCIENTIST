'''
Business Understanding:
Business Problem: When scientists do research it is becoming extremely difficult to seggregate the '3' species - Versicolor, Virginica, Setosa. 

Business Objective: Maximize Species Detection Accuracy
Business Constraint: Minimize Cost of Detection

Success Criteria: 
Business - Increase effectiveness of species detection by at least 50%
ML - Achieve an accuracy of more than 80%
Economic - Save upto $1M annually

Data Understanding:
Data
a. Sepal length
b. Sepal width
c. Petal length
d. Petal width

Target 
e. Species (Versicolor, Virginica, Setosa)

150 observations & 5 columns
'''


# Import necessary libraries
from sklearn.datasets import load_iris                    # Load Iris dataset
from sklearn.ensemble import RandomForestClassifier      # Random Forest Classifier
from sklearn.svm import LinearSVC                        # Linear Support Vector Classifier
from sklearn.linear_model import LogisticRegression      # Logistic Regression Classifier
from sklearn.preprocessing import StandardScaler         # StandardScaler for preprocessing
from sklearn.pipeline import make_pipeline               # Pipeline for preprocessing
from sklearn.ensemble import StackingClassifier          # Stacking Classifier
from sklearn.model_selection import train_test_split     # Train-test split
from sklearn import metrics                              # Metrics for model evaluation
import numpy as np                                       # Numerical computation library
import pandas as pd                                      # Data manipulation library
import pickle                                            # Pickle for model serialization

# Load the dataset
iris = load_iris()

# Create the dataframe
df_features = pd.DataFrame(data=iris.data, columns=iris.feature_names)   # Features dataframe
df_target = pd.DataFrame(data=iris.target, columns=['species'])          # Target dataframe
final = pd.concat([df_features, df_target], axis=1)                      # Concatenate features and target dataframes

X = np.array(final.iloc[:, :4])    # Extract predictors
Y = np.array(final['species'])     # Extract target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state=42)

# Define base estimators for stacking
estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
              ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))]

# Define the meta-model stacked on top of base estimators
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Fit the model on training data
stacking = clf.fit(X_train, y_train)

# Calculate accuracy
accuracy = stacking.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save the stacking model
pickle.dump(stacking, open('stacking_iris.pkl', 'wb'))

# Load the saved model
model = pickle.load(open('stacking_iris.pkl', 'rb'))

# Load test dataset
test = pd.read_csv(r'C:\stacking_classify_flask_new\iris_test.csv')

# Make predictions on the test data
pred = model.predict(test)
print("Predictions:", pred)
