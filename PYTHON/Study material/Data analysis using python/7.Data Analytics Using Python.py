# Python Packages for Data Analysis

############## numpy ##############

import numpy as np  # Imports the numpy library and assigns the alias 'np' to it.

# A list of elements in variable 'x'
x = [10, 21, 3, 14, 15, 16]
print(x)  # Prints the list 'x'.
print(type(x))  # Prints the type of 'x'.

# how to multiply the list values with 2
print(x * 2)  # Prints the list 'x' repeated twice (values of the list are repeated).

# Numpy array will help access the values
y = np.array(x)  # Creates a numpy array 'y' from the list 'x'.
print(y)  # Prints the numpy array 'y'.
print(type(y))  # Prints the type of 'y'.

print(y * 2)  # Prints the result of multiplying each element of 'y' by 2.

y > 10  # Checks which elements of 'y' are greater than 10.
y[y > 10]  # Retrieves the elements of 'y' that are greater than 10.

from numpy import random  # Imports the 'random' submodule from numpy.

x = random.randint(3, 9)  # Generates a random integer between 3 (inclusive) and 9 (exclusive) and assigns it to 'x'.

print(x)  # Prints the random integer generated.


### Memory savers

# numpy arrays share the memory instead of allocating seperate memory

a = np.arange(10)  # Create a numpy array 'a' containing numbers from 0 to 9
print(a)

b = a[::2]  # Create a new array 'b' by slicing 'a' with a step of 2
print(b)

print(np.shares_memory(a, b))  # Check whether 'a' and 'b' share the same memory
a, b  # Display both arrays 'a' and 'b'

import numpy as np  # Import the numpy library

arr = np.array([[1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10]])  # Create a 2D numpy array 'arr'

print(arr[1, 1])  # Print the element at row 0, column 1 of 'arr'
print(arr[0, 2])  # Print the element at row 0, column 2 of 'arr'

import numpy as np  # Import the numpy library again (re-importing for each code block is not necessary)

arr = np.array([1, 2, 3, 4, 5, 6, 7])  # Create a 1D numpy array 'arr'

print(arr[1:4])  # Slice elements from the beginning to index 4 (not included) of 'arr'


import numpy as np  # Import the numpy library again (re-importing for each code block is not necessary)

arr = np.array([1, 2, 3, 4, 5, 6, 7])  # Create a 1D numpy array 'arr'

print(arr[1:6:2])  # Slice elements from the beginning to index 1 to index 5 of 'arr' with a step of 2
### Masking arrays

# Create a numpy array 'a' containing numbers from 0 to 9
a = np.arange(10)
print(a)

# Create a mask for even numbers in 'a'
mask = (a%2 == 0)  # Define a mask for even numbers
print(mask)

# Apply the mask to 'a' to get only the even numbers
a[mask]


### numpy matrices

# Import the numpy library
import numpy as np

# Numpy matrices 

# Define a 2x2 matrix 'a' using string notation
a = np.matrix('1 2; 3 4')
print(a)

# Define a 2x2 matrix 'b' using list notation
b = np.matrix([[1, 2], [3, 4]])
print(b)



### numpy broadcasting

# numpy broadcasting helps when working on different sized arrays

# Create a numpy array 'a' containing numbers from 0 to 30 with a step of 10
a = np.arange(0, 40, 10)
print(a)

# Create a numpy array 'b' containing numbers 0, 1, 2
b = np.array([0, 1, 2])
print(b)

# Print shapes of arrays 'a' and 'b'
print(a, a.shape)
print(b, b.shape)

# Add a new axis to 'a'
a = a[:, np.newaxis]  # Adds one extra axis
print(a)
print('********')
print(b, '\n')
print('*********')

# Perform addition between 'a' and 'b' using broadcasting
a + b  # Addition takes place between different dimensions


# Data Structures

## Arrays/Vectors

# Import the 'array' module
from array import *

# Create an array 'array1' of type 'i' (integers) with initial values
array1 = array('i', [10, 20, 30, 40, 50])

# Iterate through elements of 'array1' and print each element
for x in array1:
    print(x)

# Display help documentation for the 'array' module
help(array)

# Insert operation 

# Import the 'array' module again (re-importing for each code block is not necessary)
from array import *

# Create an array 'array1' of type 'i' (integers) with initial values
array1 = array('i', [10, 20, 30, 40, 50])

# Insert the value 60 at index 1 in 'array1'
array1.insert(1, 60)

# Iterate through elements of 'array1' and print each element
for x in array1:
    print(x)

# Deletion operation 

# Import the 'array' module again (re-importing for each code block is not necessary)
from array import *

# Create an array 'array1' of type 'i' (integers) with initial values
array1 = array('i', [10, 20, 30, 40, 50])

# Remove the value 40 from 'array1'
array1.remove(40)

# Iterate through elements of 'array1' and print each element
for x in array1:
    print(x)

# Update operation

# Import the 'array' module again (re-importing for each code block is not necessary)
from array import *

# Create an array 'array1' of type 'i' (integers) with initial values
array1 = array('i', [10, 20, 30, 40, 50])

# Update the value at index 2 of 'array1' to 80
array1[2] = 80

# Iterate through elements of 'array1' and print each element
for x in array1:
    print(x)

## 2D Arrays

# Accessing values in a Two Dimensional Array.
# The data elements in two dimensional arrays can be accessed using two indices.
# One index referring to the main or parent array and another index referring to the position of the data element in the inner array.
# If we mention only one index then the entire inner array is printed for that index position.

# Import the 'array' module
from array import *

# Define a 2D array 'T'
T = [[11, 12, 5, 2], [15, 6, 10], [10, 8, 12, 5], [12, 15, 8, 6]]

# Print the first inner array of 'T'
print(T[0])

# Print the value at index [1][2] of 'T'
print(T[1][2])

# Inserting Values in Two Dimensional Array

# Import the 'array' module again (re-importing for each code block is not necessary)
from array import *

# Define a 2D array 'T'
T = [[11, 12, 5, 2], [15, 6, 10], [10, 8, 12, 5], [12, 15, 8, 6]]
print(T)

# Insert a new inner array at index 2 of 'T'
T.insert(2, [0, 5, 11, 13, 6])

# Print 'T' after insertion
T

# Updating Values in Two Dimensional Array

# Import the 'array' module again
from array import *

# Define a 2D array 'T'
T = [[11, 12, 5, 2], [15, 6, 10], [10, 8, 12, 5], [12, 15, 8, 6]]
print(T)

# Update the inner array at index 2 of 'T'
T[2] = [11, 9]

# Update the value at index [0][3] of 'T'
T[0][3] = 7

# Print 'T' after updating
T

# Deleting the Values in Two Dimensional Array

# Import the 'array' module again
from array import *

# Define a 2D array 'T'
T = [[11, 12, 5, 2], [15, 6, 10], [10, 8, 12, 5], [12, 15, 8, 6]]
print(T)

# Delete the inner array at index 3 of 'T'
del T[3]

# Print 'T' after deletion
T

## Matrix

# Matrix is a special case of two dimensional array where each data element is of strictly same size. So every matrix is also a two dimensional array but not vice versa. Matrices are very important data structures for many mathematical and scientific calculations.

# Matrix Example

# Import the 'array' module again
from numpy import * 

# Define a matrix 'a' with string labels and numerical values
a = array([['Mon', 18, 20, 22, 17], ['Tue', 11, 18, 21, 18],
           ['Wed', 15, 21, 20, 19], ['Thu', 11, 20, 22, 21],
           ['Fri', 18, 17, 23, 22], ['Sat', 12, 22, 20, 18],
           ['Sun', 13, 15, 19, 16]])

# Reshape 'a' into a matrix 'm' with dimensions 7x5
m = reshape(a, (7, 5))
print(m)

# Accessing Values in a Matrix

# Import the 'array' module again
from numpy import * 

# Define a matrix 'm' with string labels and numerical values
m = array([['Mon', 18, 20, 22, 17], ['Tue', 11, 18, 21, 18],
           ['Wed', 15, 21, 20, 19], ['Thu', 11, 20, 22, 21],
           ['Fri', 18, 17, 23, 22], ['Sat', 12, 22, 20, 18],
           ['Sun', 13, 15, 19, 16]])

# Print the data for Wednesday
print(m[2])

# Print the data for Friday evening
print(m[4][3])

# Print the matrix 'm'
m

# Adding the data

# Import the numpy module
from numpy import * 

# Define a matrix 'm' with string labels and numerical values
m = array([['Mon', 18, 20, 22, 17], ['Tue', 11, 18, 21, 18],
           ['Wed', 15, 21, 20, 19], ['Thu', 11, 20, 22, 21],
           ['Fri', 18, 17, 23, 22], ['Sat', 12, 22, 20, 18],
           ['Sun', 13, 15, 19, 16]])
print(m, '\n')

# Append a new row to 'm' using the 'append' function
m_r = append(m, [['Avg', 12, 15, 13, 11]], 0) # Add a row at the end
# Insert a new column to 'm' using the 'insert' function
m_c = insert(m, [5], [[1], [2], [3], [4], [5], [6], [7]], 0) # Add a column at index 5

print(m_r)  # Print the resulting matrix after appending
print('***********')
print(m_c)  # Print the resulting matrix after inserting

# Delete from matrix

# Define a matrix 'm'
m = array([['Mon', 18, 20, 22, 17], ['Tue', 11, 18, 21, 18],
           ['Wed', 15, 21, 20, 19], ['Thu', 11, 20, 22, 21],
           ['Fri', 18, 17, 23, 22], ['Sat', 12, 22, 20, 18],
           ['Sun', 13, 15, 19, 16]])

print(m)  # Print the original matrix
print('********************')
    
# Delete a row at index 2 from 'm' using the 'delete' function
n1 = delete(m, [2], 0)
print(n1)  # Print 'n1' after deleting the row
print('***********')
# Delete a column at index 2 from 'm' using the 'delete' function
n2 = delete(m, [2], 1)
n2

# Import numpy as np
import numpy as np

# Create a diagonal matrix 'z' using numpy
z = np.diag(1 + np.arange(4))
print(z)  # Print the resulting matrix
type(z)  # Display the type of 'z'

# Create a numpy array 'z' containing zeros of size 10
z = np.zeros(10)
print(z)  # Print 'z'
z[4] = 1  # Set the value at index 4 to 1
print(z)  # Print 'z' after modification

# Create a random vector 'z' of size 30 and find its mean value
z = np.random.random(30)
print(z)  # Print 'z'
m = z.mean()  # Calculate the mean value of 'z'
print(m)  # Print the mean value

# Reverse a vector (first element becomes last)
z = np.arange(50)  # Create a numpy array containing numbers from 0 to 49
print(z)  # Print 'z'
z = z[::-1]  # Reverse the order of elements in 'z'
print(z)  # Print the reversed 'z'


############## Pandas ##############
# Data Manipulation package 

# pip install pandas

# Import pandas as pd
import pandas as pd  # Importing pandas which is useful for creating dataframes

# Define lists 'a1', 'a2', and 'a3'
a1 = [1, 2, 3, 4, 4]  # List format 
a2 = [10, 11, 12, 13, 14]  # List format 
a3 = list(range(5))  # List format 

# Print 'a3'
a3

# Print 'a1', 'a2', and 'a3'
a1, a2, a3

# Creating a data frame using explicit lists
# Dataframe is a pandas object
# Contains rows and columns 
df = pd.DataFrame(columns = ["X1", "X2", "X3"])  # Create an empty dataframe with column names "X1", "X2", and "X3"
print(df)  # Print the dataframe

# Assign lists 'a1', 'a2', and 'a3' to columns "X1", "X2", and "X3" of the dataframe 'df'
df["X1"] = a1  # Convert list format into pandas series format
df["X2"] = a2  # Convert list format into pandas series format
df["X3"] = a3  # Convert list format into pandas series format
print(df)  # Print the dataframe

# Creating a data frame using explicit lists
# Index is row name
df_new = pd.DataFrame(columns = ['X1', 'X2', 'X3'], index = [101, 102, 103, 104, 105])  # Create an empty dataframe with specified index
df_new

# Assign lists 'a1', 'a2', and 'a3' to columns "X1", "X2", and "X3" of the dataframe 'df_new'
df_new["X1"] = a1  
df_new["X2"] = a2  
df_new["X3"] = a3
df_new

# Accessing columns using "." (dot) operation
df.X1  # Access column "X1"
# Accessing columns alternative way
df["X1"]  # Access column "X1"


# Accessing multiple columns : giving column names as input in list format
df[["X1", "X2"]]

# Accessing elements using ".iloc" : accessing each cell by row and column index values
df.iloc[0:3, 1] #Column can be called with only index postion when we use iloc ,here 1 is column index position

df.iloc[:,:] #to get entire data frame 
df.loc[0:2, ["X1", "X2"]] #column can be called with only names when we use loc , here [X1,X2]

# Statistics
df
df['X3'].mean()  # Calculate the mean of column 'X3'
df['X3'].median()  # Calculate the median of column 'X3'
df['X3'].mode()  # Calculate the mode of column 'X3'

df.describe()  # Generate descriptive statistics of the dataframe

# Merge operation using pandas
df1 = pd.DataFrame({"X1": [1, 2, 3], "X2": [4, 8, 12]})  # Creating DataFrame df1
df2 = pd.DataFrame({"X1": [1, 2, 3, 4], "X3": [14, 18, 112, 15]})  # Creating DataFrame df2
df1, df2
merge = pd.merge(df1, df2, on="X1")  # Merge df1 and df2 on column 'X1'
merge

# Replace index name
df = pd.DataFrame({"X1": [1, 2, 3], "X2": [4, 8, 12]})  # Creating DataFrame df
df

df.set_index("X1", inplace = True)  # Assigning index names using column 'X1'
df

# Change the column names 
df = pd.DataFrame({"X1": [1, 2, 3], "X2": [4, 8, 12]})  # Creating DataFrame df
df  = df.rename(columns = {"X1":"X3","X2":"X4"})  # Renaming columns 'X1' to 'X3' and 'X2' to 'X4'
print(df)

# Concatenation
df1 = pd.DataFrame({"X1": [1, 2, 3], "X2": [4, 8, 12]}, index=(2000, 2001, 2002))  # Creating DataFrame df1
df2 = pd.DataFrame({"X1": [4, 5, 6], "X2": [14, 16, 18]}, index=(2003, 2004, 2005))  # Creating DataFrame df2

Concatenate = pd.concat([df1, df2])  # Concatenating df1 and df2
print(Concatenate)

##########################
# Importing necessary libraries
import numpy as np
import pandas as pd

# Creating DataFrame df with columns 'grade1' and 'grade2'
x1 = [1, 2, 3, 4, 5, np.nan]
x2 = [np.nan, 11, 12, 100, np.nan, 200] 
df=pd.DataFrame()
df['grade1']=x1
df['grade2']=x2
print(df)

# Finding null values
df.isna().sum()  # Counting null values in each column
df.dropna()  # Dropping rows with null values

# Another way to create DataFrame
df = pd.DataFrame(
    {"a" : [4, 5, 6],
     "b" : [7, 8, 9],           
     "c" : [10, 11, 12]},
    index = [1, 2, 3])  # Creating DataFrame df using dictionary and specifying index
df

# Another way to create DataFrame
df = pd.DataFrame(
    [[4, 7, 10],
     [5, 8, 11],
     [6, 9, 12]],
    index = [1, 2, 3],
    columns = ['a', 'b', 'c'])  # Creating DataFrame df using list of lists and specifying column names
df

# Creating Series 'a', 'b', 'c'
a = pd.Series([50, 40, 34, 30, 22, 28, 17, 19, 20, 13, 9, 15, 10, 7, 3])
len(a)
a.plot()  # Plotting Series 'a'
a.plot(figsize = (8,6), color = 'green', title = 'line plot', fontsize = 12)  # Plotting Series 'a' with customizations
b = pd.Series([45, 22, 12, 9, 20, 34, 28, 19, 26, 38, 41, 24, 14, 32])
len(b)
c = pd.Series([25, 38, 33, 38, 23, 12, 30, 37, 34, 22, 16, 24, 12, 9])
len(c)

# Creating DataFrame 'd' with Series 'a', 'b', 'c'
d = pd.DataFrame({'a': a, 'b': b, 'c': c}, index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
d
d.plot.area(figsize = (9,6), title = 'Area plot')  # Plotting area plot for DataFrame 'd'
d.plot.area(alpha = 0.4, color = ['coral','purple','lightgreen'], figsize = (8,6), fontsize = 12)  # Plotting area plot with customizations


############## Reading Extrnal File ##############
import pandas as pd
help(pd.read_csv)
# Import data (.csv file) using pandas. We are using mba data set
mba = pd.read_csv("education.csv")

type(mba)  # Check the type of variable 'mba' (should be pandas DataFrame)
mba  # Display the DataFrame 'mba'

mba.groupby('gmat').count()  # Group 'mba' DataFrame by 'gmat' column and count occurrences of each value

mba.groupby('gmat').count()['datasrno']  # Group 'mba' DataFrame by 'gmat' column, count occurrences of each value, and then select 'datasrno' column

list(mba.groupby('gmat'))  # Convert the grouped DataFrame into a list

mba.groupby('gmat').sum().sort_values(by = 'workex')  # Group 'mba' DataFrame by 'gmat' column, sum the values for each group, and sort by 'workex' column

mba.groupby('gmat').sum().sort_values(by = 'workex', ascending = False)  # Group 'mba' DataFrame by 'gmat' column, sum the values for each group, and sort by 'workex' column in descending order

############## Descriptive Statistical Analytics / EDA ##############
# First Moment of Business Decision: Measure of Central Tendency
# Second Moment of Business Decision: Measure of Dispersion
# Third Moment of Business Decision: Measure of Asymmetry
# Fourth Moment of Business Decision: Measure of Peakedness
import pandas as pd

dir(pd)  # Display list of attributes and methods available in the pandas module

# Read data into Python
education = pd.read_csv(r"education.csv")  # Read CSV file into a DataFrame
# Education = pd.read_csv("D:\Python\education.csv")  # Alternative method to read CSV file

A = 10  # Assign integer value 10 to variable A
a = 10.1  # Assign float value 10.1 to variable a

education.info()  # Display concise summary of the DataFrame 'education'

# C:\Users\education.csv - this is windows default file path with a '\'
# C:\\Users\\education.csv - change it to '\\' to make it work in Python

# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
education.workex.mean()  # Calculate mean of the 'workex' column
education.workex.median()  # Calculate median of the 'workex' column
education.workex.mode()  # Calculate mode of the 'workex' column

# pip install numpy
from scipy import stats
stats.mode(education.workex)  # Calculate mode of the 'workex' column using SciPy's stats module

# Measures of Dispersion / Second moment business decision
education.workex.var()  # Calculate variance of the 'workex' column
education.workex.std()  # Calculate standard deviation of the 'workex' column
range = max(education.workex) - min(education.workex)  # Calculate range of the 'workex' column

# Third moment business decision
education.workex.skew()  # Calculate skewness of the 'workex' column
education.gmat.skew()  # Calculate skewness of the 'gmat' column

# Fourth moment business decision
education.workex.kurt()  # Calculate kurtosis of the 'workex' column


# Data Visualization using Python Libraries such as Matplotlib, Seaborn, Bokeh, Plotly
############## Matplotlib ##############
# Data Visualization
# pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np

education.shape  # Display the shape (dimensions) of the DataFrame 'education'

# barplot
plt.bar(height = education.gmat, x = np.arange(1, 774, 1))  # Create a bar plot with 'gmat' values

plt.hist(education.gmat)  # Create a histogram of the 'gmat' column
plt.hist(education.workex, color = 'red')  # Create a histogram of the 'workex' column with red color

help(plt.hist)  # Display help information for the plt.hist() function

plt.figure()  # Create a new figure for plotting

plt.boxplot(education.gmat)  # Create a box plot of the 'gmat' column

help(plt.boxplot)  # Display help information for the plt.boxplot() function

############## Seaborn ##############

import pandas as pd
import numpy as np
import seaborn as sns
# pip install seaborn
df = pd.read_csv("education.csv")  # Read CSV file into a DataFrame
df.dtypes  # Display data types of columns in the DataFrame 'df'

# let's find outliers in Salaries
sns.boxplot(df.gmat)  # Create a box plot of the 'gmat' column using seaborn

sns.boxplot(df.workex)  # Create a box plot of the 'workex' column using seaborn
