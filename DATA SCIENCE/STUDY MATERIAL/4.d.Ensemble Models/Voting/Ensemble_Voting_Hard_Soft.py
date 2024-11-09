'''
# CRISP-ML(Q):

Business Problem: There are a lot of assumptions in the diagnosis pertaining to cancer.
In a few cases radiologists, pathologists and oncologists go wrong in diagnosing whether 
tumor is benign (non-cancerous) or malignant (cancerous). 
Hence team of physicians want us to build an AI application which will predict with 
confidence the presence of cancer in a patient. This will serve as a compliment to the physicians.

Business Objective: Maximize Cancer Detection

Business Constraints: Minimize Treatment Cost & Maximize Patient Convenience

Success Criteria:

Business success criteria: Increase the correct diagnosis of cancer in at least 96% of patients
Machine Learning success criteria: Achieve an accuracy of atleast 98%
Economic success criteria: Reducing medical expenses will improve trust of patients and 
thereby hospital will see an increase in revenue by atleast 12%

Data Collection:

Data is collected from the hospital for 569 patients. 30 features and 1 label 
comprise the feature set. Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)
'''


# Import the required libraries
from sklearn import datasets, linear_model, neighbors, ensemble  # Import necessary modules
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.model_selection import GridSearchCV  # Import GridSearchCV for hyperparameter tuning
from sklearn.ensemble import VotingClassifier  # Import VotingClassifier for ensemble learning
import numpy as np  # Import numpy for numerical operations
import pickle  # Import pickle for model serialization

# Load the dataset
breast_cancer = datasets.load_breast_cancer()  # Load breast cancer dataset

# Input and Output
X, y = breast_cancer.data, breast_cancer.target  # Define input features (X) and target variable (y)

# Split the train and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)
# Split the dataset into train and test sets with 70% train and 30% test, maintaining class proportions and setting random seed

# Base Model 1: k-Nearest Neighbors (k-NN) with GridSearchCV
knn = neighbors.KNeighborsClassifier()  # Initialize k-NN classifier
params_knn = {'n_neighbors': np.arange(1, 25)}  # Define hyperparameter grid for k-NN
knn_gs = GridSearchCV(knn, params_knn, cv=5)  # Initialize GridSearchCV for k-NN
knn_gs.fit(X_train, y_train)  # Fit GridSearchCV to find the best k-NN model
knn_best = knn_gs.best_estimator_  # Get the best k-NN model from GridSearchCV results

# Base Model 2: Random Forest Classifier with GridSearchCV
rf = ensemble.RandomForestClassifier(random_state=0)  # Initialize Random Forest classifier
params_rf = {'n_estimators': [50, 100, 200]}  # Define hyperparameter grid for Random Forest
rf_gs = GridSearchCV(rf, params_rf, cv=5)  # Initialize GridSearchCV for Random Forest
rf_gs.fit(X_train, y_train)  # Fit GridSearchCV to find the best Random Forest model
rf_best = rf_gs.best_estimator_  # Get the best Random Forest model from GridSearchCV results

# Base Model 3: Logistic Regression with GridSearchCV
log_reg = linear_model.LogisticRegression(solver='liblinear', max_iter=2000)  # Initialize Logistic Regression classifier
C = np.logspace(1, 4, 10)  # Define regularization parameter grid for Logistic Regression
params_lr = dict(C=C)  # Define hyperparameter grid for Logistic Regression
lr_gs = GridSearchCV(log_reg, params_lr, cv=5)  # Initialize GridSearchCV for Logistic Regression
lr_gs.fit(X_train, y_train)  # Fit GridSearchCV to find the best Logistic Regression model
lr_best = lr_gs.best_estimator_  # Get the best Logistic Regression model from GridSearchCV results

# Combine all three Based models
estimators = [('knn', knn_best), ('rf', rf_best), ('log_reg', lr_best)]  # Create a list of tuples for the base estimators

# Hard/Majority Voting
# VotingClassifier with voting = "hard" parameter
ensemble_H = VotingClassifier(estimators, voting="hard")  # Initialize Hard Voting Classifier

# Fit classifier with the training data
hard_voting = ensemble_H.fit(X_train, y_train)  # Fit Hard Voting Classifier to the training data

# Save the voting classifier
pickle.dump(hard_voting, open('hard_voting.pkl', 'wb'))  # Serialize and save the Hard Voting Classifier

# Loading a saved model
model = pickle.load(open('hard_voting.pkl', 'rb'))  # Load the saved Hard Voting Classifier model using pickle

# Print model evaluation metrics
print("knn_gs.score: ", knn_best.score(X_test, y_test))  # Print the accuracy score of the best k-NN model on the test data
print("rf_gs.score: ", rf_best.score(X_test, y_test))  # Print the accuracy score of the best Random Forest model on the test data
print("log_reg.score: ", lr_best.score(X_test, y_test))  # Print the accuracy score of the best Logistic Regression model on the test data

# Hard Voting Ensembler
print("Hard Voting Ensemble Score: ", ensemble_H.score(X_test, y_test))  # Print the accuracy score of the Hard Voting Ensemble on the test data

#############################################################
# Soft Voting
# VotingClassifier with voting = "soft" parameter
ensemble_S = VotingClassifier(estimators, voting="soft")  # Initialize Soft Voting Classifier

soft_voting = ensemble_S.fit(X_train, y_train)  # Fit Soft Voting Classifier to the training data

# Save model
pickle.dump(soft_voting, open('soft_voting.pkl', 'wb'))  # Serialize and save the Soft Voting Classifier

# Load the saved model
model = pickle.load(open('soft_voting.pkl', 'rb'))  # Load the saved Soft Voting Classifier model using pickle

# Print model evaluation metrics
print("knn_gs.score: ", knn_gs.score(X_test, y_test))  # Print the accuracy score of the best k-NN model on the test data
print("rf_gs.score: ", rf_gs.score(X_test, y_test))  # Print the accuracy score of the best Random Forest model on the test data
print("log_reg.score: ", lr_gs.score(X_test, y_test))  # Print the accuracy score of the best Logistic Regression model on the test data

# Print the accuracy score of the Soft Voting Ensemble on the test data
print("Soft Voting Ensemble Score: ", ensemble_S.score(X_test, y_test))

# Output: ensemble.score:

