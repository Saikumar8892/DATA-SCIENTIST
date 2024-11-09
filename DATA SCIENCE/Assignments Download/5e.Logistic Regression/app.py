# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_engine.outliers import Winsorizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle, joblib
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, classification_report

# Load the data
data = pd.read_csv('advertising.csv')
data.info()  # Check the structure of the data

# Drop irrelevant columns if any, based on the dataset exploration
# Assuming 'Ad Topic Line' and 'City' might not be useful for prediction
data = data.drop(['Ad Topic Line', 'City'], axis=1, errors='ignore')

# Separate features (X) and target variable (Y)
X = data.drop(['Clicked_on_Ad'], axis=1)  # Features
Y = data['Clicked_on_Ad']  # Target variable

# Identify numeric features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Preprocessing pipelines for numeric and categorical data
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('winsorize', Winsorizer(capping_method='iqr', tail='both', fold=1.5)),
    ('scale', StandardScaler())
])

# Preprocessing pipeline (applies only if there are categorical features)
preprocess_pipeline = ColumnTransformer([
    ('numerical', num_pipeline, numeric_features)
])

# Fit the preprocessing pipeline
X_processed = preprocess_pipeline.fit_transform(X)
X_processed = pd.DataFrame(X_processed, columns=numeric_features)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_processed, Y, test_size=0.2, random_state=0, stratify=Y)

# Build and fit the logistic regression model using statsmodels
logit_model = sm.Logit(y_train, x_train).fit()
print(logit_model.summary())

# Save the model and preprocessing pipeline
joblib.dump(preprocess_pipeline, 'preprocess_pipeline.pkl')
pickle.dump(logit_model, open('logistic_model.pkl', 'wb'))

# Generate predictions on training data
y_train_pred = logit_model.predict(x_train)
fpr, tpr, thresholds = roc_curve(y_train, y_train_pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Predict and evaluate on training data
y_train_pred_class = (y_train_pred > optimal_threshold).astype(int)
print('Training Accuracy:', accuracy_score(y_train, y_train_pred_class))
print(classification_report(y_train, y_train_pred_class))

# Predict and evaluate on test data
y_test_pred = logit_model.predict(x_test)
y_test_pred_class = (y_test_pred > optimal_threshold).astype(int)
print('Test Accuracy:', accuracy_score(y_test, y_test_pred_class))
print(classification_report(y_test, y_test_pred_class))

# Plot the ROC Curve
plt.plot(fpr, tpr, label="AUC="+str(metrics.auc(fpr, tpr)))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()

