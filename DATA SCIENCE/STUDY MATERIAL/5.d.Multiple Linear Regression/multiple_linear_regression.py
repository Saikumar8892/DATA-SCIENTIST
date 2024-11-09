## Problem Statement
'''
The client/business deals with used cars sales.

The customers in this sector give strong preference to less-aged cars and popular brands with good resale value. This puts a very strong challenge as they only have a very limited range of vehicle options to showcase.

#### No Pre-Set Standards

`How does one determine the value of a used car?`

The Market scenario is filled with a lot of malpractices. There is no defined standards exist to determine the appropriate price for the cars, the values are determined by arbitrary methods.

The unorganized and unstructured methods are disadvantageous to the both the parties trying to strike a deal. The look and feel can be altered in used cars, but the performance cannot be altered beyond a point.


#### Revolutionizing the Used Car Industry Through Machine Learning

**Linear regression**
Linear regression is a ML model that estimates the relationship between independent variables and a dependent variable using a linear equation (straight line equation) in a multidimensional space.

**CRISP-ML(Q) process model describes six phases:**

- Business and Data Understanding
- Data Preparation (Data Engineering)
- Model Building (Machine Learning)
- Model Evaluation and Tunning
- Deployment
- Monitoring and Maintenance


**Objective(s):** Maximize the profits

**Constraints:** Maximize the customer satisfaction

**Success Criteria**

- **Business Success Criteria**: Improve the profits from anywhere between 10% to 20%

- **ML Success Criteria**: RMSE should be less than 0.15

- **Economic Success Criteria**: Second/Used cars sales delars would see an increase in revenues by atleast 20%
'''

# Load the Data and perform EDA and Data Preprocessing

# Importing necessary libraries

import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For statistical data visualization
import sidetable  # For quick summary tables
from sklearn.compose import ColumnTransformer  # For column-wise transformations
from sklearn.pipeline import Pipeline  # For building pipelines
from sklearn.impute import SimpleImputer  # For imputing missing values
from sklearn.preprocessing import MinMaxScaler  # For scaling numerical features
from sklearn.preprocessing import OneHotEncoder  # For one-hot encoding categorical features
from feature_engine.outliers import Winsorizer  # For outlier treatment
from statsmodels.stats.outliers_influence import variance_inflation_factor  # For VIF calculation
from statsmodels.tools.tools import add_constant  # For adding constant to the model
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
import statsmodels.api as sm  # For statistical models and tests
from sklearn.linear_model import LinearRegression  # For linear regression modeling
from sklearn.metrics import r2_score  # For evaluating model performance
import joblib  # For saving and loading models
import pickle  # For serializing and deserializing Python objects
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV  # For cross-validation and hyperparameter tuning
from sklearn.feature_selection import RFE  # For recursive feature elimination
from sqlalchemy import create_engine  # For database connection
from urllib.parse import quote
# Database connection
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",  # MySQL username
                               pw=quote("Deepika"),    # MySQL password
                               db="cars"))  # MySQL database name 


# Load the offline data into Database to simulate client conditions
# cars = pd.read_csv(r"CarswithEnginetype.csv")
# cars.to_sql('cars', con = engine, if_exists = 'replace', chunksize = 1000, index= False)


# Read data from MySQL database
sql = 'SELECT * FROM cars'
# sql2="show tables"
# tables = pd.read_sql_query(sql2, engine)

dataset = pd.read_sql_query(sql, engine)  # Read data from SQL database using the provided SQL query and engine

# dataset = pd.read_csv(r"CarswithEnginetype.csv")  # Alternatively, read data from a CSV file

# Descriptive Statistics and Data Distribution
dataset.describe()  # Generate descriptive statistics for the dataset

# Missing values check
dataset.isnull().any()  # Check for missing values in the dataset
dataset.info()  # Display information about the dataset

# Separating input and output variables
X = pd.DataFrame(dataset.iloc[:, 1:6])  # Extract input features from the dataset
y = pd.DataFrame(dataset.iloc[:, 0])  # Extract output variable from the dataset

# Checking for unique values
X["Enginetype"].unique()  # Get unique values of the 'Enginetype' feature
X["Enginetype"].value_counts()  # Count the occurrences of each unique value in the 'Enginetype' feature

# Build a frequency table using sidetable library
X.stb.freq(["Enginetype"])  # Use sidetable library to create a frequency table for the 'Enginetype' feature

# Separating Non-Numeric features
categorical_features = X.select_dtypes(include=['object']).columns  # Select non-numeric (categorical) features
print(categorical_features)  # Print the names of categorical features

# Separating Numeric features
numeric_features = X.select_dtypes(exclude=['object']).columns  # Select numeric features
print(numeric_features)  # Print the names of numeric features



# Outlier Analysis

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True

# Box plot to visualize the distribution and identify outliers for each numeric feature in the dataset
X.plot(kind='box', subplots=True, sharey=False, figsize=(25, 18))  # Create a box plot for each numeric feature, with separate subplots

'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''

# Increase spacing between subplots
plt.subplots_adjust(wspace = 0.75)  # Adjust the spacing between subplots to improve visualization

plt.show()  # Display the box plots

# Imputation strategy for numeric columns
num_pipeline = Pipeline([('impute', SimpleImputer(strategy = 'mean')),
                         ('scale', MinMaxScaler()),
                         ('winsorize', Winsorizer(capping_method = 'iqr', tail='both', fold=1.5))])

# Encoding categorical to numeric variable
categ_pipeline = Pipeline([('impute', SimpleImputer(strategy = 'most_frequent')),
                           ('label', OneHotEncoder(sparse = False, drop = 'first'))])

# Using ColumnTransfer to transform the columns of an array or pandas DataFrame. This estimator allows different columns or column subsets of the input to be transformed separately and the features generated by each transformer will be concatenated to form a single feature space.
preprocess_pipeline = ColumnTransformer([('numerical', num_pipeline, numeric_features), 
                                         ('categorical', categ_pipeline, categorical_features)])

clean =  preprocess_pipeline.fit(X)   # Fit the preprocess pipeline to the input features

# Save the encoding model
joblib.dump(clean, 'preprocessed')  # Save the preprocessed pipeline for future use

Clean_data = pd.DataFrame(clean.transform(X))  # Transform the categorical data using OneHotEncoder and create a DataFrame with the encoded data

# To get feature names for Categorical columns after preprocessing
Clean_data.columns = clean.get_feature_names_out(input_features=X.columns)  # Assign meaningful column names to the encoded DataFrame
Clean_data.info()  # Display information about the cleaned data DataFrame




####################
# Multivariate Analysis

# Plot pairplot for visualizing relationships between variables
# Original Data Exploration (assuming 'dataset' is a DataFrame containing your data)

sns.pairplot(dataset)  # Create a pair plot to visualize relationships between features in the original data

# Calculate Correlation Matrix
orig_df_cor = Clean_data.corr()  # Calculate the correlation matrix 

# Display Correlation Matrix
print(orig_df_cor)  # Display the correlation matrix (showing the strength and direction of linear relationships)

# Identify Collinear Pairs (based on correlation values)
# Based on the correlation matrix you might find highly correlated features:
#  - VOL - WT: 0.999 (very strong positive correlation)
#  - HP - SP : 0.973 (very strong positive correlation)
# High correlation suggests potential collinearity, which can negatively affect model performance.

# Correlation Matrix Heatmap Visualization
dataplot = sns.heatmap(orig_df_cor, annot=True, cmap="YlGnBu")  # Create a heatmap visualization of the correlation matrix
                                                               #  - 'annot=True' displays correlation values on the heatmap
                                                               #  - 'cmap="YlGnBu"' sets the colormap

# Enhanced Heatmap with Masking (optional)
mask = np.triu(np.ones_like(orig_df_cor, dtype=bool))  # Create a mask to hide redundant upper triangle in the heatmap
sns.heatmap(orig_df_cor, annot=True, mask=mask, vmin=-1, vmax=1)  # Create a heatmap with masking
plt.title('Correlation Coefficient Of Predictors')  # Set a title for the heatmap
plt.show()  # Display the enhanced heatmap

# Build a Baseline Linear Regression Model

# Add a constant term (intercept) to the clean data (assuming 'clean_data' is a DataFrame)
P = add_constant(Clean_data)

# Build a vanilla linear regression model (Ordinary Least Squares) using statsmodels
basemodel = sm.OLS(y, P).fit()  # 'y' is the target variable, 'P' is the data with constant term

# Summarize the model results
basemodel.summary()  # Print a summary of the model, including coefficients, p-values, R-squared, etc.


# High p-values of coefficients indicate insignificance due to collinearity

# Identify the variable with the highest collinearity using Variance Inflation Factor (VIF)
# Addressing Collinearity and Influential Observations

# Calculate Variance Inflation Factors (VIF)
vif = pd.Series([variance_inflation_factor(P.values, i) for i in range(P.shape[1])], index=P.columns)
print(vif)  # Display VIF values for each feature

# Identify Feature with High VIF (assuming threshold is > 5 for high collinearity)
# Based on VIF values, a feature might have a high VIF (e.g., index 3).
# This suggests potential collinearity with other features.

# Drop Feature with Highest VIF numerical__WT = 96.932596
clean_data1 = Clean_data.drop('numerical__WT', axis=1)  # Drop the feature with the highest VIF (assuming WT in this case)

# Build a Model on the Reduced Dataset
basemode2 = sm.OLS(y, clean_data1).fit()
basemode2.summary()  # Print the model summary for the reduced dataset

# Check for Influential Observations
sm.graphics.influence_plot(basemode2)  # Create influence plots to identify potentially influential observations

# Handle Influential Observations (optional)
# Based on the influence plots, you might identify influential observations (e.g., index 76 and 78).
# These observations can potentially skew the model.

# Remove Influential Observations and Build Model on Updated Dataset
clean_data1_new = clean_data1.drop(clean_data1.index[[76, 78]])  # Drop identified influential observations
y_new = y.drop(y.index[[76, 78]])  # Drop corresponding target values
basemode3 = sm.OLS(y_new, clean_data1_new).fit()
basemode3.summary()  # Print the model summary for the updated dataset with influential observations removed

# Split Data into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(clean_data1_new, y_new, test_size=0.2, random_state=0)
#  - X_train: Features for training
#  - X_test: Features for testing
#  - Y_train: Target variable for training
#  - Y_test: Target variable for testing
#  - test_size: Proportion of data for testing (0.2 = 20%)
#  - random_state: Ensures reproducibility (set for consistent splits)

# Build the Final Model (without cross-validation for simplicity)
model = sm.OLS(Y_train, X_train).fit()
model.summary()  # Print the model summary for the final model


# Evaluate Model Performance on Training and Testing Data (without Cross-Validation)

# Training Data Performance
ytrain_pred = model.predict(X_train)  # Predict target values for training data using the fitted model
r_squared_train = r2_score(Y_train, ytrain_pred)  # Calculate R-squared (coefficient of determination) for training data
                                                  # - R-squared measures the proportion of variance explained by the model
train_resid = Y_train.MPG - ytrain_pred  # Calculate residuals (differences between actual and predicted values) for training data
train_rmse = np.sqrt(np.mean(train_resid * train_resid))  # Calculate Root Mean Squared Error (RMSE) for training data
                                                        # - RMSE measures the average magnitude of the errors

# Testing Data Performance
y_pred = model.predict(X_test)  # Predict target values for testing data using the fitted model
r_squared = r2_score(Y_test, y_pred)  # Calculate R-squared (coefficient of determination) for testing data
test_resid = Y_test.MPG - y_pred  # Calculate residuals (differences between actual and predicted values) for testing data
test_rmse = np.sqrt(np.mean(test_resid * test_resid))  # Calculate Root Mean Squared Error (RMSE) for testing data

# Cross-Validation for Model Selection and Evaluation (using KFold)
lm = LinearRegression()  # Create a linear regression model object
folds = KFold(n_splits=5, shuffle=True, random_state=100)  # Define KFold cross-validation with 5 splits, shuffling, and fixed random state
scores = cross_val_score(lm, X_train, Y_train, scoring='r2', cv=folds)  # Perform KFold cross-validation to get R-squared scores on each fold

# Model Building with Cross-Validation and Feature Selection (Recursive Feature Elimination)
folds = KFold(n_splits=5, shuffle=True, random_state=100)  # Define KFold cross-validation again
hyper_params = [{'n_features_to_select': list(range(1, 9))}]  # Define hyperparameter grid for RFE (number of features to select)
lm.fit(X_train, Y_train)  # Fit the linear regression model on the entire training data (needed for RFE)
rfe = RFE(lm)  # Create a Recursive Feature Elimination object using the linear regression model
model_cv = GridSearchCV(estimator=rfe, param_grid=hyper_params,  # Create a GridSearchCV object
                        scoring='r2', cv=folds, verbose=1, return_train_score=True)  # - estimator: RFE object
                                                                                # - param_grid: Hyperparameter grid
                                                                                # - scoring: R-squared evaluation metric
                                                                                # - cv: KFold cross-validation
                                                                                # - verbose: Print progress information
                                                                                # - return_train_score: Include training scores
model_cv.fit(X_train, Y_train)  # Fit the GridSearchCV object with RFE and cross-validation
cv_results = pd.DataFrame(model_cv.cv_results_)  # Convert cross-validation results to a DataFrame


# Plotting Cross-Validation Results

plt.figure(figsize=(16, 6))  # Create a figure with desired size
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])  # Plot average test scores across different numbers of features
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])  # Plot average training scores across different numbers of features
plt.xlabel('number of features')  # Label the x-axis
plt.ylabel('r-squared')  # Label the y-axis
plt.title("Optimal Number of Features")  # Set the plot title
plt.legend(['test score', 'train score'], loc='upper left')  # Add a legend

# Saving the Best Model

pickle.dump(model_cv.best_estimator_, open('mpg.pkl', 'wb'))  # Save the best model (RFE with optimal number of features) as a pickle file

# Loading Test Data and Applying Preprocessing Steps (assuming you have these functions saved)

data = pd.read_csv(r"Cars_test.csv")  # Load test data

# Assuming 'meanimpute', 'winsor', 'minmax', and 'encoding' are functions or objects for preprocessing steps
model1 = pickle.load(open('mpg.pkl', 'rb'))  # Load the best model
impute = joblib.load('preprocessed')  # Load the imputer object
 # Load the encoding object (e.g., one-hot encoder)

# Apply preprocessing steps (assuming these are the steps used during training)

clean3 = pd.DataFrame(impute.transform(data), columns=impute.get_feature_names_out(input_features=data.columns))  # Apply encoding (e.g., one-hot encoding)



# Drop feature identified earlier (assuming WT was dropped during training)
clean_data1 = clean3.drop(clean3[['numerical__WT']], axis=1)

# Make Predictions Using the Best Model
prediction = pd.DataFrame(model1.predict(clean_data1), columns=['MPG_pred'])  # Predict MPG for the test data using the best model
print(prediction)  # Display the predictions

