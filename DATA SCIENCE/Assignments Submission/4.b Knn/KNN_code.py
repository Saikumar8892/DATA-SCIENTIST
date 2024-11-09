# IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sweetviz as sv
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from feature_engine.outliers import Winsorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import sklearn.metrics as skmet
import joblib
import pickle
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine, text
from urllib.parse import quote 
from getpass import getpass


# IMPORTING DATASET
data = pd.read_csv(r"Glass.csv")

# CREATING ENGINE TO CONNECT DATABASE
user_name = 'root'
database = 'knn'
your_password = getpass()
engine = create_engine(f'mysql+pymysql://{user_name}:%s@localhost:3306/{database}' % quote(f'{your_password}'))

data.to_sql('glass', con = engine, if_exists = 'replace', index = False)


# SELECTING THE ENTIRE DATA FROM DATABASE
sql = text("select * from glass")
data = pd.read_sql_query(sql, engine)
data

data.info() # No null values
decs = data.describe()
data.Type.value_counts() # 6 class problem

# AUTO EDA
eda = sv.analyze(data)
eda.show_html()

# CHECKING FOR DUPLICATES
duplicate = data.duplicated() # Returns Boolean Series denoting duplicate rows.
duplicate
sum(duplicate) # Found 1 duplicate

# DROPPING DUPLICATES
data1 = data.drop_duplicates()

# SEPERATING INPUTS AND OUTPUT
X = data.iloc[ :, 0:9] # Predictors
Y = data['Type'] # Target

# Separating Numeric and Non-Numeric columns
numeric_features = X.select_dtypes(exclude = ['object']).columns
numeric_features
categorical_features = X.select_dtypes(include=['object']).columns
categorical_features # there is no categorical features

num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean')), ('scale', MinMaxScaler())])
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features)])
imp_scale = preprocessor.fit(X)
joblib.dump(imp_scale, 'imp_scale')
num_data = pd.DataFrame(imp_scale.transform(X), columns = imp_scale.get_feature_names_out())

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



winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = list(num_data.iloc[:,0:7].columns))

outlier = winsor.fit(num_data[list(num_data.iloc[:,0:7].columns)])

# Save the winsorizer model 
joblib.dump(outlier, 'winsor')

num_data[list(num_data.iloc[:,0:7].columns)] = outlier.transform(num_data[list(num_data.iloc[:,0:7].columns)])
####### 
num_data.iloc[:,0:7].plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 

'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''

# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()



# SPLITTING THE DATA INTO TRAIN & TEST
X_train, X_test, Y_train, Y_test = train_test_split(num_data, Y, test_size = 0.2, random_state = 0)
X_train.shape
X_test.shape

# MODEL BUILDING
knn = KNeighborsClassifier(n_neighbors = 3)
KNN = knn.fit(X_train, Y_train)

# PREDICTING ON TEST DATA
pred_test = KNN.predict(X_test)  
pred_test

# CROSS TABLE FOR TEST DATA
pd.crosstab(Y_test, pred_test, rownames = ['Actual'], colnames = ['Predictions']) 

# ACCURACY SCORE FOR TEST DATA
print(skmet.accuracy_score(Y_test, pred_test))

# PREDICTING ON TRAIN DATA
pred_train = KNN.predict(X_train)  
pred_train

# CROSS TABLE FOR TRAIN DATA
pd.crosstab(Y_train, pred_train, rownames = ['Actual'], colnames = ['Predictions']) 

# ACCURACY SCORE FOR TRAIN DATA
print(skmet.accuracy_score(Y_train, pred_train))

# FINDING THE BEST VALUE OF 'K'
acc = []

for i in range(3, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    diff = train_acc - test_acc
    acc.append([diff, train_acc, test_acc])

acc

# TRAIN DATA ACCURACY PLOT
plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], "ro-")

# TEST DATA ACCURACY PLOT
plt.plot(np.arange(3, 50, 2), [i[2] for i in acc], "bo-")

plt.show()
k_range = list(range(3, 50, 2))
param_grid = dict(n_neighbors = k_range)
  
# HYPERPARAMETER TUNING
grid = GridSearchCV(KNN, param_grid, cv = 5, scoring = 'accuracy', return_train_score = False, verbose = 1)
KNN_new = grid.fit(X_train, Y_train) 
print(KNN_new.best_params_)

accuracy = KNN_new.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )

pred = KNN_new.predict(X_test)
pred

knn_best = KNN_new.best_estimator_

# SAVING THE MODEL
pickle.dump(knn_best, open('knn.pkl', 'wb'))
