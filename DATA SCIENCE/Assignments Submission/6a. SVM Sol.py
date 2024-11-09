import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the provided dataset
data = pd.read_csv('SalaryData_Train.csv')

# Display the first few rows and get basic info to understand the structure
data_head = data.head()
data_info = data.info()
data_head, data_info
# Check for any missing values in the dataset
missing_values = data.isnull().sum()

# Check for unique values in categorical columns to decide on encoding/cleaning
categorical_columns = data.select_dtypes(include=['object']).columns
unique_values = {col: data[col].unique() for col in categorical_columns}

missing_values, unique_values

# Remove leading/trailing whitespaces from string columns
data[categorical_columns] = data[categorical_columns].apply(lambda x: x.str.strip())

# Encode the target variable 'Salary' as binary (<=50K: 0, >50K: 1)
label_encoder = LabelEncoder()
data['Salary'] = label_encoder.fit_transform(data['Salary'])

# One-hot encode other categorical features
data_encoded = pd.get_dummies(data, columns=categorical_columns[:-1], drop_first=True)  # All except 'Salary'

# Scale numerical features
scaler = StandardScaler()
numerical_columns = ['age', 'educationno', 'capitalgain', 'capitalloss', 'hoursperweek']
data_encoded[numerical_columns] = scaler.fit_transform(data_encoded[numerical_columns])

# Split data into training and testing sets
X = data_encoded.drop('Salary', axis=1)
y = data_encoded['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Outputting processed dataset info and shapes for confirmation
data_encoded_info = data_encoded.info()
X_train_shape, X_test_shape, y_train_shape, y_test_shape = X_train.shape, X_test.shape, y_train.shape, y_test.shape

data_encoded_info, X_train_shape, X_test_shape, y_train_shape, y_test_shape

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_svm_model = grid_search.best_estimator_

y_pred = best_svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

