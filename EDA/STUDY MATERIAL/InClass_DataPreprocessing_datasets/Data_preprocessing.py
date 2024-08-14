############# Data Pre-processing ##############

################ Type casting #################

# Importing the pandas library for data manipulation and analysis
import pandas as pd  

# Reading data from a CSV file named "ethnic diversity.csv" located at "C:/Data/"
data = pd.read_csv(r"D:/New materials/EDA/InClass_DataPreprocessing_datasets/ethnic diversity.csv")  

# Displaying the data types of each column in the DataFrame
data.dtypes  

'''
EmpID is Integer - Python automatically identify the data types by interpreting the values. 
As the data for EmpID is numeric Python detects the values as int64.

From measurement levels prespective the EmpID is a Nominal data as it is an identity for each employee.

If we have to alter the data type which is defined by Python then we can use astype() function

'''

# Getting help on the astype method of pandas DataFrame
help(data.astype)

# Converting the 'EmpID' column from 'int64' to 'str' (string) type
data.EmpID = data.EmpID.astype('str')

# Displaying the data types after converting 'EmpID' column
data.dtypes

# Converting the 'Zip' column from its current type to 'str' (string) type
data.Zip = data.Zip.astype('str')

# Displaying the data types after converting 'Zip' column
data.dtypes

# For practice:
# Convert data types of columns from:

# Converting the 'Salaries' column from 'float64' to 'int64' type
data.Salaries = data.Salaries.astype('int64')

# Displaying the data types after converting 'Salaries' column
data.dtypes

# Converting the 'age' column from 'int' to 'float32' type
data.age = data.age.astype('float32')

# Displaying the data types after converting 'age' column
data.dtypes


##############################################
### Identify duplicate records in the data ###
# Importing the pandas library for data manipulation and analysis
import pandas as pd  

# Reading data from a CSV file named "mtcars_dup.csv" located at "C:/Data/"
data = pd.read_csv(r"D:/New materials/EDA/InClass_DataPreprocessing_datasets/mtcars_dup.csv")

# Getting help on the duplicated method of pandas DataFrame
help(data.duplicated)

# Finding duplicate rows in the DataFrame and storing the result in a Boolean Series
duplicate = data.duplicated()  # Returns Boolean Series denoting duplicate rows.

# Displaying the Boolean Series indicating duplicate rows
duplicate

# Counting the total number of duplicate rows
sum(duplicate)

# Finding duplicate rows in the DataFrame and keeping the last occurrence of each duplicated row
duplicate = data.duplicated(keep='last')
duplicate

# Finding all duplicate rows in the DataFrame
duplicate = data.duplicated(keep=False)
duplicate

# Removing duplicate rows from the DataFrame and storing the result in a new DataFrame
data1 = data.drop_duplicates() # Returns DataFrame with duplicate rows removed.

# Removing duplicate rows from the DataFrame and keeping the last occurrence of each duplicated row
data1 = data.drop_duplicates(keep='last')

# Removing all duplicate rows from the DataFrame
data1 = data.drop_duplicates(keep=False)


# Duplicates in Columns
# We can use correlation coefficient values to identify columns which have duplicate information

# Importing the pandas library for data manipulation and analysis
import pandas as pd  

# Reading data from a CSV file named "Cars.csv" located at "C:/Data/"
cars = pd.read_csv(r"D:/New materials/EDA/InClass_DataPreprocessing_datasets/Cars.csv")

# Correlation coefficient
'''
Ranges from -1 to +1. 
Rule of thumb says |r| > 0.85 is a strong relation
'''
# Calculating the correlation matrix for the columns in the DataFrame
cars.corr()

'''We can observe that the correlation value for HP and SP is 0.973 and VOL and WT is 0.999 
& hence we can ignore one of the variables in these pairs.
'''


################################################
############## Outlier Treatment ###############
# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns

# Reading data from a CSV file named "ethnic diversity.csv" located at "C:/Data/"
df = pd.read_csv(r"D:/New materials/EDA/InClass_DataPreprocessing_datasets/ethnic diversity.csv")

# Displaying the data types of each column in the DataFrame
df.dtypes

# Creating a box plot to visualize the distribution and potential outliers in the 'Salaries' column
sns.boxplot(df.Salaries)

# Creating a box plot to visualize the distribution and potential outliers in the 'age' column
sns.boxplot(df.age)
# No outliers in age column

# Detection of outliers in the 'Salaries' column using the Interquartile Range (IQR) method
IQR = df['Salaries'].quantile(0.75) - df['Salaries'].quantile(0.25)

# Calculating the lower and upper limits for outlier detection based on IQR
lower_limit = df['Salaries'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Salaries'].quantile(0.75) + (IQR * 1.5)

############### 1. Remove (let's trim the dataset) ################
# Trimming Technique
# Let's flag the outliers in the dataset

# Creating a boolean array indicating whether each value in the 'Salaries' column is an outlier
outliers_df = np.where(df.Salaries > upper_limit, True, np.where(df.Salaries < lower_limit, True, False))

# Filtering the DataFrame to include only rows where 'Salaries' column contains outliers
df_out = df.loc[outliers_df, ]

# Filtering the DataFrame to exclude rows containing outliers
df_trimmed = df.loc[~(outliers_df), ]

# Displaying the shape of the original DataFrame and the trimmed DataFrame
df.shape, df_trimmed.shape

# Creating a box plot to visualize the distribution of 'Salaries' in the trimmed dataset
sns.boxplot(df_trimmed.Salaries)

############### 2. Replace ###############
# Replace the outliers by the maximum and minimum limit
# Creating a new column 'df_replaced' in the DataFrame with values replaced by upper or lower limit if they are outliers
df['df_replaced'] = pd.DataFrame(np.where(df['Salaries'] > upper_limit, upper_limit, np.where(df['Salaries'] < lower_limit, lower_limit, df['Salaries'])))

# Creating a box plot to visualize the distribution of 'df_replaced' column
sns.boxplot(df.df_replaced)


############### 3. Winsorization ###############
# pip install feature_engine   # install the package
# Importing the Winsorizer class from the feature_engine.outliers module
from feature_engine.outliers import Winsorizer

# Defining the Winsorizer model with IQR method
# Parameters:
# - capping_method: 'iqr' specifies the Interquartile Range (IQR) method for capping outliers
# - tail: 'both' indicates that both tails of the distribution will be capped
# - fold: 1.5 specifies the multiplier to determine the range for capping outliers based on IQR
# - variables: ['Salaries'] specifies the column(s) in the DataFrame to apply the Winsorizer to
winsor_iqr = Winsorizer(capping_method='iqr', 
                        tail='both', 
                        fold=1.5, 
                        variables=['Salaries'])

# Fitting the Winsorizer model to the 'Salaries' column and transforming the data
df_s = winsor_iqr.fit_transform(df[['Salaries']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Creating a box plot to visualize the distribution of 'Salaries' after applying Winsorizer with IQR method
sns.boxplot(df_s.Salaries)

# Defining the Winsorizer model with Gaussian method
# Parameters:
# - capping_method: 'gaussian' specifies the Gaussian method for capping outliers
# - tail: 'both' indicates that both tails of the distribution will be capped
# - fold: 3 specifies the number of standard deviations to determine the range for capping outliers based on Gaussian method
# - variables: ['Salaries'] specifies the column(s) in the DataFrame to apply the Winsorizer to
winsor_gaussian = Winsorizer(capping_method='gaussian', 
                             tail='both', 
                             fold=3,
                             variables=['Salaries'])

# Fitting the Winsorizer model to the 'Salaries' column and transforming the data
df_t = winsor_gaussian.fit_transform(df[['Salaries']])

# Creating a box plot to visualize the distribution of 'Salaries' after applying Winsorizer with Gaussian method
sns.boxplot(df_t.Salaries)


# Define the model with percentiles:
# Default values
# Right tail: 95th percentile
# Left tail: 5th percentile

# Defining the Winsorizer model with quantiles method
# Parameters:
# - capping_method: 'quantiles' specifies the quantiles method for capping outliers
# - tail: 'both' indicates that both tails of the distribution will be capped
# - fold: 0.05 specifies the proportion of data to be excluded from the lower and upper ends of the distribution (5th and 95th percentiles)
# - variables: ['Salaries'] specifies the column(s) in the DataFrame to apply the Winsorizer to
winsor_percentile = Winsorizer(capping_method='quantiles',
                               tail='both', 
                               fold=0.05, 
                               variables=['Salaries'])

# Fitting the Winsorizer model to the 'Salaries' column and transforming the data
df_p = winsor_percentile.fit_transform(df[['Salaries']])

# Creating a box plot to visualize the distribution of 'Salaries' after applying Winsorizer with quantiles method
sns.boxplot(df_p.Salaries)


##############################################
#### zero variance and near zero variance ####

# Importing the pandas library for data manipulation and analysis
import pandas as pd  

# Reading data from a CSV file named "ethnic diversity.csv" located at "C:/Data/"
df = pd.read_csv(r"D:/New materials/EDA/InClass_DataPreprocessing_datasets/ethnic diversity.csv")

# Displaying the data types of each column in the DataFrame
df.dtypes
# If the variance is low or close to zero, then a feature is approximately constant and will not improve the performance of the model.
# In that case, it should be removed. 

# Select only numeric columns
numeric_columns = df.select_dtypes(include=np.number)

# Calculating the variance of each numeric variable in the DataFrame
numeric_columns.var()

# Checking if the variance of each numeric variable is equal to 0 and returning a boolean Series
numeric_columns.var() == 0 

# Checking if the variance of each numeric variable along axis 0 (columns) is equal to 0 and returning a boolean Series
numeric_columns.var(axis=0) == 0 


#############
# Discretization

# Importing the pandas library for data manipulation and analysis
import pandas as pd

# Reading data from a CSV file named "ethnic diversity.csv" located at "C:/Data/"
data = pd.read_csv(r"D:/New materials/EDA/InClass_DataPreprocessing_datasets/ethnic diversity.csv")

# Displaying the first few rows of the DataFrame
data.head()

# Displaying the last few rows of the DataFrame
data.tail()

# Displaying information about the DataFrame, including the data types of each column and memory usage
data.info()

# Generating descriptive statistics of the DataFrame, including count, mean, standard deviation, minimum, maximum, and quartile values
data.describe()

# Binarizing the 'Salaries' column into two categories ('Low' and 'High') based on custom bins
data['Salaries_new'] = pd.cut(data['Salaries'], 
                              bins=[min(data.Salaries), data.Salaries.mean(), max(data.Salaries)],
                              labels=["Low", "High"])

# Counting the number of occurrences of each category in the 'Salaries_new' column
data.Salaries_new.value_counts()


''' We can observe that the total number of values are 309. This is because one of the value has become NA.
This happens as the cut function by default does not consider the lowest (min) value while discretizing the values.
To over come this issue we can use the parameter 'include_lowest' set to True.
'''

# Binarizing the 'Salaries' column into two categories ('Low' and 'High') based on custom bins
# Parameters:
# - bins: Custom bins defined by the minimum salary value, mean salary value, and maximum salary value
# - include_lowest: Whether to include the lowest edge of the bins in the intervals
# - labels: Labels assigned to the resulting categories
data['Salaries_new1'] = pd.cut(data['Salaries'], 
                              bins=[min(data.Salaries), data.Salaries.mean(), max(data.Salaries)], 
                              include_lowest=True,
                              labels=["Low", "High"])

# Counting the number of occurrences of each category in the 'Salaries_new1' column
data.Salaries_new1.value_counts()

#########
# Importing the matplotlib library for creating plots
import matplotlib.pyplot as plt

# Creating a bar plot to visualize the distribution of 'Salaries_new1' categories
plt.bar(x=range(310), height=data.Salaries_new1)

# Creating a histogram to visualize the distribution of 'Salaries_new1' categories
plt.hist(data.Salaries_new1)

# Creating a box plot to visualize the distribution of 'Salaries_new1' categories
plt.boxplot(data.Salaries_new1)

# Discretization into multiple bins based on quartiles
data['Salaries_multi'] = pd.cut(data['Salaries'], 
                              bins=[min(data.Salaries), 
                                    data.Salaries.quantile(0.25),
                                    data.Salaries.mean(),
                                    data.Salaries.quantile(0.75),
                                    max(data.Salaries)], 
                              include_lowest=True,
                              labels=["P1", "P2", "P3", "P4"])

# Counting the number of occurrences of each category in the 'Salaries_multi' column
data.Salaries_multi.value_counts()

# Counting the number of occurrences of each category in the 'MaritalDesc' column
data.MaritalDesc.value_counts()


##################################################
################## Dummy Variables ###############
# methods:
    # get dummies
    # One Hot Encoding
    # Label Encoding
    # Ordinal Encoding
# Importing the pandas library for data manipulation and analysis
import pandas as pd
# Importing the numpy library for numerical computing
import numpy as np

# Reading data from a CSV file named "ethnic diversity.csv" located at "C:/Data/"
df = pd.read_csv(r"D:/New materials/EDA/InClass_DataPreprocessing_datasets/ethnic diversity.csv")

# Displaying the names of all columns in the DataFrame
df.columns 

# Displaying the shape of the DataFrame (number of rows and columns)
df.shape 

# Displaying the data types of each column in the DataFrame
df.dtypes

# Displaying concise summary of the DataFrame including non-null counts and data types
df.info()

# Dropping the columns 'Employee_Name', 'EmpID', 'Zip' from the DataFrame and storing the result in a new DataFrame
df1 = df.drop(['Employee_Name', 'EmpID', 'Zip'], axis=1)

# Dropping the columns 'Employee_Name', 'EmpID', 'Zip' from the DataFrame inplace (modifying original DataFrame)
df.drop(['Employee_Name', 'EmpID', 'Zip'], axis=1, inplace=True)

# Creating dummy variables for categorical columns in the DataFrame and storing the result in a new DataFrame
# The pd.get_dummies() function returns True and False values by default, but by applying .astype('int64') to its output DataFrame, you convert these Boolean values to integers, resulting in 1s and 0s.
df_new = pd.get_dummies(df).astype('int64') 

# Creating dummy variables for categorical columns in the DataFrame and dropping the first category of each column
df_new_1 = pd.get_dummies(df, drop_first=True).astype('int64')

##### One Hot Encoding works
# Displaying the names of all columns in the DataFrame
df.columns 

# Selecting specific columns and updating the DataFrame with the selected columns
df = df[['Salaries', 'age', 'Position', 'State', 'Sex',
         'MaritalDesc', 'CitizenDesc', 'EmploymentStatus', 'Department', 'Race']]

# Extracting the 'Salaries' column as a pandas Series
a = df['Salaries']

# Extracting the 'Salaries' column as a DataFrame
b = df[['Salaries']]

# Importing the OneHotEncoder class from the sklearn.preprocessing module
from sklearn.preprocessing import OneHotEncoder

# Creating an instance of the OneHotEncoder
enc = OneHotEncoder(sparse_output=False) # initializing method 
# setting sparse_output=False explicitly instructs the OneHotEncoder to return a dense array instead of a sparse matrix.

# Transforming the categorical columns (from Position column onwards) into one-hot encoded format and converting to DataFrame
enc_df = pd.DataFrame(enc.fit_transform(df.iloc[:, 2:]), columns= enc.get_feature_names_out(input_features=df.iloc[:, 2:].columns))

#######################
# Label Encoder
# Label Encoding is typically applied to a single column or feature at a time, meaning it operates on one-dimensional data.
# Importing the LabelEncoder class from the sklearn.preprocessing module
from sklearn.preprocessing import LabelEncoder

# Creating an instance of the LabelEncoder
labelencoder = LabelEncoder()

# Data Split into Input and Output variables
# X contains the features (independent variables), excluding the last column
X = df.iloc[:, :9]
# y contains the target variable (dependent variable), which is the last column
y = df.iloc[:, 9]

# Transforming the 'Sex' column into numerical labels using LabelEncoder
X['Sex'] = labelencoder.fit_transform(X['Sex'])

# Transforming the 'MaritalDesc' column into numerical labels using LabelEncoder
X['MaritalDesc'] = labelencoder.fit_transform(X['MaritalDesc'])

# Transforming the 'CitizenDesc' column into numerical labels using LabelEncoder
X['CitizenDesc'] = labelencoder.fit_transform(X['CitizenDesc'])

########################
# Ordinal Encoding
# Importing the OrdinalEncoder class from the sklearn.preprocessing module
from sklearn.preprocessing import OrdinalEncoder
# Ordinal Encoding can handle multiple dimensions or features simultaneously.
oe = OrdinalEncoder()
# Data Split into Input and Output variables
# X contains the features (independent variables), excluding the last column
X = df.iloc[:, :9]
# y contains the target variable (dependent variable), which is the last column
y = df.iloc[:, 9]

X[['Sex', 'MaritalDesc', 'CitizenDesc']] = oe.fit_transform(X[['Sex', 'MaritalDesc', 'CitizenDesc']])


#################### Missing Values - Imputation ###########################
# Importing the necessary libraries
import numpy as np
import pandas as pd

# Loading the modified ethnic dataset from a CSV file located at "C:/Data/modified ethnic.csv"
df = pd.read_csv(r'D:/New materials/EDA/InClass_DataPreprocessing_datasets/modified ethnic.csv') # for doing modifications

# Checking for the count of missing values (NA's) in each column of the DataFrame
df.isna().sum()

# Create an imputer object that fills 'Nan' values
# Mean and Median imputer are used for numeric data (Salaries)
# Mode is used for discrete data (ex: Position, Sex, MaritalDesc)

# Importing SimpleImputer from the sklearn.impute module
from sklearn.impute import SimpleImputer

# Mean Imputer: Replacing missing values in the 'Salaries' column with the mean value
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df["Salaries"] = pd.DataFrame(mean_imputer.fit_transform(df[["Salaries"]]))
df["Salaries"].isna().sum()  # Checking for any remaining missing values in 'Salaries'

# Median Imputer: Replacing missing values in the 'age' column with the median value
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df["age"] = pd.DataFrame(median_imputer.fit_transform(df[["age"]]))
df["age"].isna().sum()  # Checking for any remaining missing values in 'age' 

# Mode Imputer: Replacing missing values in the 'Sex' and 'MaritalDesc' columns with the most frequent value
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df["Sex"] = pd.DataFrame(mode_imputer.fit_transform(df[["Sex"]]))
df["MaritalDesc"] = pd.DataFrame(mode_imputer.fit_transform(df[["MaritalDesc"]]))
df.isnull().sum()  # Checking for any remaining missing values

# Constant Value Imputer: Replacing missing values in the 'Sex' column with a constant value 'F'
constant_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='F')
df["Sex"] = pd.DataFrame(constant_imputer.fit_transform(df[["Sex"]]))

# Random Imputer: Replacing missing values in the 'age' column with random samples from the same column
from feature_engine.imputation import RandomSampleImputer

random_imputer = RandomSampleImputer(['age'])
df["age"] = pd.DataFrame(random_imputer.fit_transform(df[["age"]]))
df["age"].isna().sum()  # Checking for any remaining missing values in 'age'


#####################
# Normal Quantile-Quantile Plot

# Importing pandas library for data manipulation
import pandas as pd

# Reading data from a CSV file named "education.csv" located at "C:/Data/"
education = pd.read_csv(r"D:/New materials/EDA/InClass_DataPreprocessing_datasets/education.csv")

# Importing scipy.stats module for statistical functions
import scipy.stats as stats
# Importing pylab module for creating plots
import pylab

# Checking whether the 'gmat' data is normally distributed using a Q-Q plot
stats.probplot(education.gmat, dist="norm", plot=pylab)

# Checking whether the 'workex' data is normally distributed using a Q-Q plot
stats.probplot(education.workex, dist="norm", plot=pylab)

# Importing numpy module for numerical computations
import numpy as np

# Transformation to make 'workex' variable normal by applying logarithmic transformation
stats.probplot(np.log(education.workex), dist="norm", plot=pylab)

# Importing seaborn and matplotlib.pyplot for plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Original data
prob = stats.probplot(education.workex, dist=stats.norm, plot=pylab)

# Transforming the 'workex' data using Box-Cox transformation and saving the lambda value
fitted_data, fitted_lambda = stats.boxcox(education.workex)

# Creating subplots
fig, ax = plt.subplots(1, 2)

# Plotting the original and transformed data distributions
sns.distplot(education.workex, hist=False, kde=True,
             kde_kws={'shade': True, 'linewidth': 2},
             label="Non-Normal", color="green", ax=ax[0])

sns.distplot(fitted_data, hist=False, kde=True,
             kde_kws={'shade': True, 'linewidth': 2},
             label="Normal", color="green", ax=ax[1])

# Adding legends to the subplots
plt.legend(loc="upper right")

# Rescaling the subplots
fig.set_figheight(5)
fig.set_figwidth(10)

# Printing the lambda value used for transformation
print(f"Lambda value used for Transformation: {fitted_lambda}")

# Transformed data
prob = stats.probplot(fitted_data, dist=stats.norm, plot=pylab)


# Yeo-Johnson Transform

'''
We can apply it to our dataset without scaling the data.
It supports zero values and negative values. It does not require the values for 
each input variable to be strictly positive. 

In Box-Cox transform the input variable has to be positive.
'''

# Importing pandas library for data manipulation
import pandas as pd
# Importing stats module from scipy library for statistical functions
from scipy import stats

# Importing seaborn and matplotlib.pyplot for plotting
import seaborn as sns
import matplotlib.pyplot as plt
# Importing pylab module for creating plots
import pylab

# Read data from a CSV file named "education.csv" located at "C:/Data/"
education = pd.read_csv(r"D:/New materials/EDA/InClass_DataPreprocessing_datasets/education.csv")

# Original data
# Checking whether the 'workex' data is normally distributed using a Q-Q plot
prob = stats.probplot(education.workex, dist=stats.norm, plot=pylab)

# Importing transformation module from feature_engine library
from feature_engine import transformation

# Set up the Yeo-Johnson transformer for 'workex' variable
tf = transformation.YeoJohnsonTransformer(variables='workex')

# Transforming the 'workex' variable using Yeo-Johnson transformation
edu_tf = tf.fit_transform(education)

# Transformed data
# Checking whether the transformed 'workex' data is normally distributed using a Q-Q plot
prob = stats.probplot(edu_tf.workex, dist=stats.norm, plot=pylab)



####################################################
######## Standardization and Normalization #########

# Importing pandas library for data manipulation
import pandas as pd
# Importing numpy library for numerical computations
import numpy as np

# Reading data from a CSV file named "mtcars.csv" located at "D:/Data/"
data = pd.read_csv(r"D:/New materials/EDA/InClass_DataPreprocessing_datasets/mtcars.csv")

# Generating descriptive statistics of the original data
a = data.describe()

# Importing StandardScaler from the sklearn.preprocessing module
from sklearn.preprocessing import StandardScaler

# Initialise the StandardScaler
scaler = StandardScaler()

# Scaling the data using StandardScaler
df = scaler.fit_transform(data)

# Converting the scaled array back to a DataFrame
dataset = pd.DataFrame(df)

# Generating descriptive statistics of the scaled data
res = dataset.describe()


# Normalization
''' Alternatively we can use the below function'''
# Importing MinMaxScaler from the sklearn.preprocessing module
from sklearn.preprocessing import MinMaxScaler

# Initializing the MinMaxScaler
minmaxscale = MinMaxScaler()

# Scaling the data using MinMaxScaler
df_n = minmaxscale.fit_transform(df)

# Converting the scaled array back to a DataFrame
dataset1 = pd.DataFrame(df_n)

# Generating descriptive statistics of the scaled data
res1 = dataset1.describe()


### Normalization
# Load dataset from a CSV file named "ethnic diversity.csv" located at "D:/Data/"
ethnic1 = pd.read_csv(r"D:/New materials/EDA/InClass_DataPreprocessing_datasets/ethnic diversity.csv")

# Displaying column names of the dataset
ethnic1.columns

# Dropping columns 'Employee_Name', 'EmpID', 'Zip' from the dataset
ethnic1.drop(['Employee_Name', 'EmpID', 'Zip'], axis=1, inplace=True)

# Generating descriptive statistics of the original dataset
a1 = ethnic1.describe()

# Generating dummy variables for categorical columns in the dataset and dropping the first category of each column

ethnic = pd.get_dummies(ethnic1, drop_first=True).astype(int)
# Generating descriptive statistics of the dataset with dummy variables
a2 = ethnic.describe()

### Normalization function - Custom Function
# Range converts values to range between 0 and 1
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x

# Applying normalization function to the dataset
df_norm = norm_func(ethnic)

# Generating descriptive statistics of the normalized dataset
b = df_norm.describe()


''' Alternatively we can use the below function'''
# Importing MinMaxScaler from the sklearn.preprocessing module
from sklearn.preprocessing import MinMaxScaler

# Initializing the MinMaxScaler
minmaxscale = MinMaxScaler()

# Scaling the dataset using MinMaxScaler
ethnic_minmax = minmaxscale.fit_transform(ethnic)

# Converting the scaled array back to a DataFrame
df_ethnic = pd.DataFrame(ethnic_minmax)

# Generating descriptive statistics of the dataset after Min-Max scaling
minmax_res = df_ethnic.describe()


'''Robust Scaling
Scale features using statistics that are robust to outliers'''

# Importing RobustScaler from the sklearn.preprocessing module
from sklearn.preprocessing import RobustScaler

# Initializing the RobustScaler
robust_model = RobustScaler()

# Scaling the dataset using RobustScaler
df_robust = robust_model.fit_transform(ethnic)

# Converting the scaled array back to a DataFrame
dataset_robust = pd.DataFrame(df_robust)

# Generating descriptive statistics of the dataset after Robust scaling
res_robust = dataset_robust.describe()
