import pandas as pd  # Pandas library for data manipulation and analysis.
import numpy as np  # Numpy library for numerical computations and handling arrays.
# Libraries for visualization
import matplotlib.pyplot as plt  # Matplotlib for creating plots.
import seaborn as sns  # Seaborn for enhanced statistical visualizations.
# Scikit-learn preprocessing and transformation tools
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler  # Encodes categorical data and scales numerical data.
from sklearn.compose import ColumnTransformer  # Applies transformers to specified columns in a DataFrame.
# For calculating multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor  # Assesses multicollinearity using VIF.
# Model building and evaluation libraries
from sklearn.model_selection import train_test_split  # Splits the dataset into training and testing sets.
from sklearn.linear_model import LinearRegression  # Implements linear regression modeling.
from sklearn.metrics import mean_squared_error, r2_score  # Evaluates model performance with RMSE and R^2 metrics.
# Cross-validation and feature selection tools
from sklearn.model_selection import cross_val_score, KFold  # Performs cross-validation.
from sklearn.feature_selection import RFE  # Recursive Feature Elimination for feature selection.
from sklearn.model_selection import GridSearchCV  # Optimizes parameters through grid search with cross-validation.
# Load the dataset
data = pd.read_csv('50_Startups.csv')  # Reads the dataset into a DataFrame.
data.head()  # Displays the first few rows of the data to understand its structure.
data.info()  # Shows summary information about the dataset (e.g., data types, non-null counts).

# Check for missing values and summarize descriptive statistics for numerical features.
missing_values = data.isnull().sum()  # Counts missing values in each column.
descriptive_stats = data.describe()  # Provides summary statistics for numerical columns.
# Set the visual style
sns.set(style="whitegrid")  # Configures Seaborn to use a clean, white-grid style.

# Plot distributions of numerical features to understand their spread and identify outliers.
plt.figure(figsize=(16, 10))
for i, col in enumerate(['R&D Spend', 'Administration', 'Marketing Spend', 'Profit'], 1):
    plt.subplot(2, 2, i)
    sns.histplot(data[col], kde=True)  # Plots the histogram with KDE for each feature.
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()  # Displays the plots for univariate analysis.
# Visualize pairwise relationships between numerical features and the target variable.
sns.pairplot(data, x_vars=['R&D Spend', 'Administration', 'Marketing Spend'], y_vars='Profit', height=4, aspect=1, kind='scatter')
plt.suptitle('Bivariate Analysis with Profit', y=1.05)  # Adds a title for clarity.
plt.show()
# Calculate the correlation matrix, excluding the 'State' column since it's categorical.
corr_matrix = data.drop(columns='State').corr()  # Computes correlation between numerical features.

# Plot the heatmap for the correlation matrix.
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')  # Annotates cells with correlation coefficients.
plt.title("Correlation Matrix of Numerical Features")  # Sets title for the heatmap.
plt.show()
# Define a preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), ['R&D Spend', 'Administration', 'Marketing Spend']),  # Scales numeric features.
    ('cat', OneHotEncoder(drop='first'), ['State'])  # Encodes categorical 'State' column.
])

# Define features and target variable
X = data.drop(columns='Profit')  # Drops target column to keep only input features.
y = data['Profit']  # Sets the target variable (Profit).

# Fit and transform the data using the pipeline
X_processed = preprocessor.fit_transform(X)  # Applies transformations to features.
# Convert processed features into a DataFrame and add a constant column for intercept
X_vif = pd.DataFrame(X_processed)  # Converts transformed features to a DataFrame for VIF calculation.
X_vif['const'] = 1  # Adds an intercept column to calculate VIF properly.

# Calculate VIF for each feature
vif_data = pd.DataFrame()  # Creates a DataFrame to store VIF values.
vif_data['Feature'] = X_vif.columns  # Lists feature names.
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]  # Computes VIF.
print(vif_data)  # Outputs VIF values for multicollinearity assessment.
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=0)  # Uses 80-20 split.

# Train a linear regression model
model = LinearRegression()  # Initializes a Linear Regression model.
model.fit(X_train, y_train)  # Fits the model on training data.

# Predictions and evaluation on both training and testing sets
y_train_pred = model.predict(X_train)  # Predicts on training data.
y_test_pred = model.predict(X_test)  # Predicts on testing data.

# Calculate performance metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))  # Computes RMSE for training set.
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))  # Computes RMSE for testing set.
train_r2 = r2_score(y_train, y_train_pred)  # Computes R^2 for training set.
test_r2 = r2_score(y_test, y_test_pred)  # Computes R^2 for testing set.
print("Training RMSE:", train_rmse, "Testing RMSE:", test_rmse)  # Outputs RMSE values.
print("Training R2:", train_r2, "Testing R2:", test_r2)  # Outputs R^2 values.
# Define cross-validation strategy and RFE for feature selection
folds = KFold(n_splits=5, shuffle=True, random_state=100)  # Sets up 5-fold cross-validation.
rfe = RFE(model)  # Initializes RFE for feature selection.

# Use GridSearch to find the optimal number of features
param_grid = {'n_features_to_select': list(range(1, X_processed.shape[1] + 1))}  # Range for feature selection.
grid_search = GridSearchCV(estimator=rfe, param_grid=param_grid, scoring='r2', cv=folds)  # Defines GridSearchCV.
grid_search.fit(X_train, y_train)  # Fits grid search on training data.

print("Best number of features:", grid_search.best_params_)  # Outputs the optimal number of features.
print("Best R-squared score:", grid_search.best_score_)  # Outputs the best R^2 score from cross-validation.
