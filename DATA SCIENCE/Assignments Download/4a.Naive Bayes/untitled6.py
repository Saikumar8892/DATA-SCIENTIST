import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your training data
train_data = pd.read_csv('SalaryData_Train.csv')

# Preprocess the data
def preprocess_data(data):
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data, label_encoders

# Preprocess training data
train_data_processed, label_encoders = preprocess_data(train_data.copy())

# Separate features and target
X_train = train_data_processed.drop(columns='Salary')
y_train = train_data_processed['Salary']

# Initialize and train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(nb_model, 'naive_bayes_salary_model.pkl')
print("Model saved as naive_bayes_salary_model.pkl")
