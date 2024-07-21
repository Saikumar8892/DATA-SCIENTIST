# File Handling

import os
os.getcwd()# Get the current working directory

os.chdir('D:\\Data')# Change the current working directory to 'D:\\Data'

# file handling #
# the open function takes two parameters filename and model #

# there are 4 different methods for opening a file #
# "r" - reads a default value
# "a" - opens the file for append and creates a file if it does not exist
# "w" - write - opens a file for writing
# "x" - creates a file 

# syntax #
# to open a file or read a file 
# Open the file "demo1.txt" in read mode
f = open("demo1.txt") 

# Print the file object
print(f)

# Open the file "demo1.txt" using a context manager
with open("demo1.txt") as file:
    # Read the contents of the file
    xy = file.read()

# Print the contents of the file
print(xy)

# Open the file "demo1.txt" in write mode
f = open("demo1.txt", "w")

# Write some text to the file
f.write("Adding new lines")

# Close the file
f.close()

# Open the file "demo1.txt" again
f = open("demo1.txt")

f.readline()   # one line at a time  --> fist line

print(f.readline())   # --> second line
print(f.readline())   # --> third line
print(f.readline())   # --> forth line

f = open("demo1.txt", mode = 'r')
print(f.readlines()) # outputs each line separated with comma and all the lines at once
f.close()

f = open("demo1.txt") 
f.read(5) # reads first 5 characters

# write to an existing file #
f = open("demo1.txt", "a")   # create or append
f.write("\nNow the file has more content!")
f.close()

f = open("demo1.txt", "r")
f.read() #observe one extra line is added

# creating a new file #
f = open("demo2.txt", "x")   # create
f.write(' New file')
f.close()

f = open('demo2.txt', 'r')
print(f.read())

f.close()

# Delete a File
# To delete a file, you must import the OS module, and run its os.remove() function:
import os
# Remove the file "demo2.txt"
os.remove("demo2.txt")

# Open the file "education.csv" in read mode
f = open(r'D:\Data\education.csv','r')

# Read and print the contents of the file
print(f.read())

# Close the file
f.close()


# Exception Handling

x = eval(input("enter the 1st value = "))

# try: and except: are keywords for exceptional handling, try: provide the code ; except: except the error with a proper statement

# Ex - 1:
# Try block to handle potential errors
try:
    # Accept user input for the first value and evaluate it
    x = eval(input("enter the 1st value = "))
    # Accept user input for the second value and evaluate it
    y = eval(input("enter the 2nd value = "))
    # Perform addition operation
    results = x + y
    # Print the result
    print(results)
# Exception block to handle any error that occurs within the try block
except:
    # Print a message if an error occurs
    print("please enter a valid number")

# This line intentionally raises a ZeroDivisionError
6/0

#Ex - 2: 
try:
    # Accept user input for the first value and evaluate it
    x = eval(input("enter the 1st value = "))
    # Accept user input for the second value and evaluate it
    y = eval(input("enter the 2nd value = "))
    # Perform division operation
    results = x/y
    # Print the result
    print("final results = ", results)
# Exception block to handle specific errors
except(ZeroDivisionError):
    # Print a message if a ZeroDivisionError occurs
    print("please enter a non-zero value for the divisor")
except(NameError):
    # Print a message if a NameError occurs
    print("Please enter valid number")
except(TypeError):
    # Print a message if a TypeError occurs
    print("Please enter both same type")


## In Python, keywords 'else' and 'finally' can also be used along with 
# the 'try' and 'except' clauses while the 'except' block is executed

## if the exception occurs inside the 'try' block, the 'else' block gets 
# processed if the 'try' block is found to be exception free.
# Try block to execute the code that may raise an exception
try:
    # Print a message indicating the execution of the try block
    print("try block")
    # Accept user input for the first number and convert it to an integer
    x = int(input('Enter a number: '))
    # Accept user input for the second number and convert it to an integer
    y = int(input('Enter another number: '))
    # Perform division operation
    z = x / y
# Exception block to handle ZeroDivisionError
except ZeroDivisionError:
    # Print a message if a ZeroDivisionError occurs
    print("except ZeroDivisionError block")
    print("Division by 0 not accepted")
# Else block to execute if no exception occurs
else:
    # Print a message indicating the execution of the else block
    print("else block")
    # Print the result of the division operation
    print("Division = ", z)
# Finally block to execute cleanup code, regardless of whether an exception occurred or not
finally:
    # Print a message indicating the execution of the finally block
    print("finally block")
    # Reset variables x and y to 0
    x = 0
    y = 0
# Print a message indicating the end of the try-except-else-finally structure
print ("Out of try, except, else, and finally blocks.")

## Built-in Exceptions

## There are plenty of built-in exceptions in Python that are raised when 
# corresponding errors occur. 
## We can view all the built-in exceptions using the built-in local() function
# as follows:
print(dir(locals()['__builtins__']))


## User Defined Exceptions

# Accept user input for the number
x = int(input('Enter the number: '))
# Check if the entered number is less than 18
if x < 18:
    # Raise an Exception with a custom error message if the condition is met
    raise Exception("Sorry, You are not eligible for voting, register once you become 18 years old")
else:
    # Print the entered number if the condition is not met
    print('you entered: ', x)
