"""Desicion Tree!! -360DIGITMG

Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems.
Decision Tree uses a flowchart like a tree structure to show the predictions that result from a series of feature-based splits. It starts with a root node and ends with a decision made by leaves.

Random Forest
A random forest is a machine learning technique that’s used to solve regression and classification problems. It utilizes ensemble learning, which is a technique that combines many classifiers to provide solutions to complex problems.

A random forest algorithm consists of many decision trees. The ‘forest’ generated by the random forest algorithm is trained through bagging or bootstrap aggregating.
Decision Tree problem 1: A cloth manufacturing company is interested to know about the segment or attributes contributing to high sale. Approach - A decision tree and random forest model can be built with target variable 'Sale' (we will first convert it into categorical variable) & all other variables will be independent in the analysis.
Business Objective: Predict whether a sales are High or Not 
"""
import pandas as pd # for dataset manipulation
import numpy as np # # for arithmetic calculations
import matplotlib.pyplot as plt # for visualisation
import seaborn as sns # for visualisation

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import joblib
import pickle

# Importing company dataset using pandas
company = pd.read_csv('ClothCompany_Data.csv')

company.head()
company.columns
company
# Data Description: We will be trying to predict the sales of product. In this data set, a single observation represents a location where product are sold.

# Sales - Unit sales (in thousands) at each location

# CompPrice - Price charged by competitor at each location

# Income - Community income level (in thousands of dollars)

# Advertising - Local advertising budget for company at each location (in thousands of dollars)

# Population - Population size in region (in thousands)

# Price - Price company charges for product at each site

# ShelveLoc - A factor with levels Bad, Good and Medium indicating the quality of the shelving location for the product at each 
# site

# Age - Average age of the local population

# Education - Education level at each location

# Urban - A factor with levels No and Yes to indicate whether the store is in an urban or rural location

# US - A factor with levels No and Yes to indicate whether the store is in the US or not
company.info()
 # 3 variables are categorical - Urban, US and ShelveLoc
company.ShelveLoc.value_counts(normalize = True)
company.Urban.value_counts(normalize = True)
company.US.value_counts(normalize = True)
company.describe()
## EDA
# sweetviz
##########

# pip install sweetviz
import sweetviz
my_report = sweetviz.analyze([company, "data"])

my_report.show_notebook('Report1.html')
## Pre-Processing
# Missing values
company.isnull().sum()
# There are no missing values in our dataset
# we have to convert sales data to categorical by binning
# we will alot top 33% to high category
company['Sales'] = np.where(company['Sales'] <= company['Sales'].quantile(.67), 'Not High', 'High')
# Input and Output Split
predictors = company.loc[:, company.columns!="Sales"] # All row and all columns except sales column
target = company["Sales"] 
target
#Separating Numeric and Non-Numeric columns
numeric_features = predictors.select_dtypes(exclude = ['object']).columns
categorical_features = predictors.select_dtypes(include=['object']).columns
num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean')), ('scale', MinMaxScaler())])
cat_pipeline = Pipeline(steps = [('encoding', OneHotEncoder())])
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features),('categorical', cat_pipeline, categorical_features)])
imp_enc_scale = preprocessor.fit(predictors)
#### Save the imputation model using joblib
joblib.dump(imp_enc_scale, 'imp_enc_scale')
num_data = pd.DataFrame(imp_enc_scale.transform(predictors), columns = imp_enc_scale.get_feature_names_out())

num_data
# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True

num_data.iloc[:,0:7].plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 

'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''


# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()
num_data.iloc[:,0:7].columns
winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['num__CompPrice', 'num__Income', 'num__Advertising', 'num__Population',
       'num__Price', 'num__Age', 'num__Education'])
outlier = winsor.fit(num_data[['num__CompPrice', 'num__Income', 'num__Advertising', 'num__Population',
       'num__Price', 'num__Age', 'num__Education']])
# Save the winsorizer model 
joblib.dump(outlier, 'winsor')
num_data[['num__CompPrice', 'num__Income', 'num__Advertising', 'num__Population',
           'num__Price', 'num__Age', 'num__Education']] = outlier.transform(num_data[['num__CompPrice', 'num__Income', 'num__Advertising', 'num__Population',
                                                                                       'num__Price', 'num__Age', 'num__Education']])
num_data.iloc[:,0:7].plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 

# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()
# label_Sales 
target = target.map({'Not High':0,'High':1})
target
## Model Building
# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(num_data, target, test_size = 0.25, random_state= 42, stratify = target )
from sklearn.tree import DecisionTreeClassifier as DT
model = DT(criterion = 'entropy')
model.fit(x_train, y_train)
# Prediction on Test Data
preds = model.predict(x_test)
pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predictions'])
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test, model.predict(x_test))
# Prediction on Train Data
preds = model.predict(x_train)
pd.crosstab(y_train, preds, rownames = ['Actual'], colnames = ['Predictions'])
# Train Data Accuracy 
accuracy_score(y_train, model.predict(x_train))
#Test data accuracy is 0.77 and train data accuracy is 1 so the model is overfitting
# let us try random forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)
rf_clf.fit(x_train, y_train)
confusion_matrix(y_test, rf_clf.predict(x_test))
# Test Data Accuracy 
accuracy_score(y_test, rf_clf.predict(x_test))
#Test data accuracy for decision tree was 0.77 and for Random forest is 0.78, accuracy has been increased by 2%.
confusion_matrix(y_train, rf_clf.predict(x_train))
# Train Data Accuracy 
accuracy_score(y_train, rf_clf.predict(x_train))
#Test data accuracy is 0.78 and train data accuracy is 1 so the model is overfitting
## Hyperparameter Tuning for Decission Tree
#As the model is overfitting, we do some hyperparameter tuning to get the desired result
# create a dictionary of all hyperparameters to be experimented
param_grid = { 'criterion':['gini','entropy'], 'max_depth': np.arange(3, 15)}

# Decision tree model
dtree_model = DT()

# GridsearchCV with cross-validation to perform experiments with parameters set
dtree_gscv = GridSearchCV(dtree_model, param_grid, cv = 5, scoring = 'accuracy', return_train_score = False, verbose = 1)
# Train
dtree_gscv.fit(x_train, y_train)
# The best set of parameter values
dtree_gscv.best_params_
# Model with best parameter values
DT_best = dtree_gscv.best_estimator_
# Prediction on Test Data
#pip install graphviz
preds1 = DT_best.predict(x_test)
preds1
pd.crosstab(y_test, preds1, rownames = ['Actual'], colnames= ['Predictions'])
# Accuracy
print(accuracy_score(y_test, preds1))
import os
import graphviz
from sklearn import tree
from sklearn.tree import export_graphviz
predictors = list(num_data.columns)
type(predictors)
num_data, target
dot_data = tree.export_graphviz(DT_best, filled = True, 
                                rounded = True,
                                feature_names = predictors,
                                class_names = ['HIgh', "Not High"],
                                out_file = None)
graph = graphviz.Source(dot_data)
graph
# Prediction on Train Data

preds_train = DT_best.predict(x_train)
preds_train
# Confusion Matrix
pd.crosstab(y_train, preds_train, rownames = ['Actual'], colnames = ['Predictions'])
# Accuracy

print(accuracy_score(y_train, preds_train))
## Hyperparameter Tuning for RandomForestClassifier
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid)

rf_Model = RandomForestClassifier()

# GridsearchCV with cross-validation to perform experiments with parameters set
rf_gscv = GridSearchCV(rf_Model, param_grid, cv = 5, scoring = 'accuracy', return_train_score = False, verbose = 1)
# Train
rf_gscv.fit(x_train, y_train)
# The best set of parameter values
rf_gscv.best_params_
# Model with best parameter values
rf_best = rf_gscv.best_estimator_
rf_best
# Prediction on Test Data

preds1 = rf_best.predict(x_test)
preds1
pd.crosstab(y_test, preds1, rownames = ['Actual'], colnames= ['Predictions'])
# Accuracy
print(accuracy_score(y_test, preds1))
# Prediction on Train Data

preds_train = rf_best.predict(x_train)
preds_train
# Confusion Matrix
pd.crosstab(y_train, preds_train, rownames = ['Actual'], colnames = ['Predictions'])
# Accuracy

print(accuracy_score(y_train, preds_train))
### Save the Best Model with pickel library
pickle.dump(rf_best, open('rf.pkl', 'wb'))
pickle.dump(DT_best, open('dt.pkl', 'wb'))
import os
os.getcwd()