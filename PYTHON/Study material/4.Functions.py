# Functions

# Example for advantages of functions
# Imagine there are 100 lines of code created based on requirement without using functions.

# Print the string "Hello Function.Python" to the console.
print("Hello Function.Python") 

# Print the string "Hello Function.Python" to the console.
print("Hello Function.Python") 

# Print the string "Hello Function.Python" to the console.
print("Hello Function.Python") 

# Print the string "Hello Function.Python" to the console.
print("Hello Function.Python") 



def sayHello():
    print('Hello World!') # Block belonging to the function
# End of function #

sayHello()

# Not defined any value but defined an empty function 
def hello_func():              
    pass

# Call the function hello_func().
hello_func()


# Simple function 
def hello(name, age, sal):                 
    # Print a greeting message along with name, age, and salary information.
    print("hi", name, "your age:", age, "your salary:", sal)

# Call the hello function with arguments "Sharat", 25, and 50000.
hello("Sharat", 25, 50000)


# Formal and actual arguments

# Function definition
def add(a, b):
    return a + b 


# function call
add(2, 3)


# Formal arguments are identifiers used in the function definition to represent 
# corresponding actual arguments.

# Actual arguments are values (or variables / expressions) that are used inside
# the parentheses of a function call.


# Positional arguments
def rectangleArea(width, height):
    # Calculate the area of a rectangle using width and height.
    return width * height

# Print the area of a rectangle with width 1 and height 2.
print(rectangleArea(width=1, height=2))

# A positional argument is a name that is not followed by an equal sign (=) and default value.

def printMax(a, b):
    # Check if a is greater than b.
    if a < b:
        # Print a is maximum if it is greater.
        print(a, 'is maximum')
    # Check if a is equal to b.
    elif a == b:
        # Print a is equal to b if they are equal.
        print(a, 'is equal to', b)
    else:
        # Print b is maximum if a is not greater than b.
        print(b, 'is maximum')

# Call printMax function with arguments 3 and 4.
printMax(3, 4) 

def say(message, times=4):
    # Print the message times the specified number of times.
    print(message * times)

# Call say function with the message 'Hello' and default times value of 1.
say('Hello')

# Call say function with the message 'World ' and times value of 5.
say('World ', 5)



def func(a, b=5, c=10):
    # Print values of a, b, and c.
    print('a is', a, 'and b is', b, 'and c is', c)

# Call func with arguments 3 and 7, using default values for b and c.
func(3, 7)
# Call func with arguments 25 and using default value for b, and explicitly setting c to 24.
func(25, c = 24)
# Call func with argument a set to 100, and explicitly setting c to 50, using default value for b.
func(c = 50, a = 100)


x = 50
def func(x):
    # Print the value of x.
    print('x is', x)
    # Change the value of local variable x.
    x = 2
    # Print the changed value of local variable x.
    print('Changed local x to', x)

# Print the value of x outside of the function.
print('x is still', x)


x = 50
def func(x):
    # Print the value of x.
    print('x is', x)
    # Change the value of local variable x.
    x = 2
    # Print the changed value of local variable x.
    print('Changed local x to', x)

# Call func with the global variable x as an argument.
func(x)
# Print the value of x outside of the function.
print('x is still', x)

# Variable arguments
# *args for variable number of arguments 
def myFun(*argv):  
    # Iterate through all arguments passed to the function.
    for arg in argv:  
        # Print each argument.
        print(arg) 

# Call myFun with multiple arguments.
myFun('Hello', 'Welcome', 'to', '360digitmg')  


# Use **kwargs to pass the variable keyword arguments to the function 

def intro(**data):
    # Print the data type of the argument.
    print("\nData type of argument:", type(data))

    # Iterate through the key-value pairs in the data dictionary.
    for key, value in data.items():
        # Print each key-value pair.
        print("{} is {}".format(key, value))

# Call intro function with a dictionary of key-value pairs as arguments.
intro(Firstname="Sita", Lastname="Sharma", Age=22, Phone=1234567890)
intro(Firstname="John", Lastname="Wood", Email="johnwood@nomail.com", Country="Wakanda", Age=25, Phone=9876543210)


def total(initial, *numbers, **keywords):
    # Initialize count with the initial value.
    count = initial
    
    # Iterate through the positional arguments (numbers) and add them to count.
    for number in numbers:
        count += number
    
    # Iterate through the keyword arguments (keywords) and add their values to count.
    for key in keywords:
        count += keywords[key]
    
    # Return the total count.
    return count

# Call the total function with initial value 10, positional arguments 1, 2, 3, and keyword arguments vegetables=50, fruits=100.
print(total(10, 1, 2, 3, vegetables=50, fruits=100))

## Function call stack
### locals(), globals()
x = 5  # Define a global variable x with value 5.

def new():
    # Access the global variable x and print its value.
    print('number is', x)

# Call the new function.
new()

# Print the global variable x outside the function.
print(x)

x = 5  # Define a global variable x with value 5.

def new():
    # Define a local variable x with value 10.
    x = 10
    # Print the local variable x.
    print('number is', x)

# Call the new function.
new()

# Print the global variable x outside the function.
print(x)


age = 23 # Age Global variable
name = "Somesh" # Name Global Variable
places = ["Nagpur", "Mumbai", "Pune"] # Places Global Variable


def local(): # Function declaration  
    age = 1 # Age Local Variable
    name = 'Aayush' # Name Local Variable
    print("%s is %i years old and lives in %s." % (name, age, places[0])) # Function Definition for Local Variable
    # In the above line of code name and age variables are Local and place is Global Variable 
    return

local() # Function calls Local() Variables
print("##############")

print("%s is %i years old and lives in %s." % (name, age, places[2])) # Calling efinition for Global Variable
#In the above line of code name, age and place are Global Variables 


## Declaring global variable within function
x = 5

def new():
    global x  # Declare x as global within the function.
    x = 10    # Assign a new value to the global variable x.
    print('number is', x)

# Call the new function, which modifies the global variable x.
new()

# Print the modified value of the global variable x.
print(x)



## Recursion

# A defined function can call itself.

# Define a function called fact which calculates the factorial of a given number N.
def fact(N):
    # Check if the input N is negative.
    if N < 0:
        # If N is negative, print an error message.
        print("Input the positive number")
    # Check if the input N is zero.
    elif N == 0:
        # If N is zero, print the factorial of 0 which is 1.
        print(f'Factorial of {N}! is {1}')
    else:
        # If N is positive, initialize the variable fact to 1.
        fact = 1
        # Iterate through the range from 1 to N (inclusive).
        for i in range(1, N + 1):
            # Multiply fact by each value in the range to calculate the factorial.
            fact *= i
        # Print the factorial of N.
        print(f'Factorial of {N}! is {fact}')

# Call the fact function with argument 5.
fact(5)


# Redefine the function fact to calculate factorial recursively.
def fact(N):
    # Check if the input N is negative.
    if N < 0:
        # If N is negative, print an error message.
        print("Input the positive number")
    # Check if the input N is zero.
    elif N == 0:
        # If N is zero, print the factorial of 0 which is 1.
        print(f'Factorial of {N}! is {1}')
    # Check if the input N is 1.
    elif N == 1:
        # If N is 1, return 1 (base case for recursion).
        return 1
    else:
        # If N is greater than 1, recursively call the fact function with N-1 and multiply the result by N.
        return (N * fact(N-1))

# Call the fact function with argument 5.
fact(5)




# List Comprehensions and Readability 

# Build a list of Unicode codepoints from a string in ordinary way 
# Define a string containing various currency symbols.
symbols = '$¢£¥€¤'

# Initialize an empty list to store Unicode values of the symbols.
codes = []

# Iterate over each symbol in the symbols string.
for symbol in symbols:
    # Append the Unicode value of the current symbol to the codes list using ord() function.
    codes.append(ord(symbol))

# Print the list of Unicode values.
print(codes)


# Alternative way to perform the above Unicode conversion using list comprehension.
# Define the symbols string again.
symbols = '$¢£¥€'

# Use list comprehension to iterate over each symbol in the symbols string and convert them to Unicode.
codes = [ord(symbol) for symbol in symbols]

# Print the list of Unicode values.
print(codes)


# Create a new list containing odd numbers from 0 to 11 using list comprehension.
new_list = [i for i in range(12) if i % 2 == 1]

# Print the new list.
print(new_list)


## Dictionary Comprehension

# Let us see how the same problem can be solved using a for loop and dictionary comprehension
# We want to create a new dictionary where the key is a number divisible by 2 in a range of 0-10 and it's value is the square of the number.
# Define a range of numbers from 0 to 9.
numbers = range(10)

# Print the range object.
print(numbers)

# Initialize an empty dictionary to store square values of even numbers.
new_dict_for = {}

# Add values to 'new_dict_for' using 'for' loop.
for n in numbers:
    # Check if the current number is even.
    if n % 2 == 0:
        # Calculate the square of the current number and store it as value in the dictionary.
        new_dict_for[n] = n ** 2

# Print the resulting dictionary.
# Print the dictionary created using 'for' loop.
print(new_dict_for)

# Use dictionary comprehension to create a dictionary with square values of even numbers.
new_dict_comp = {n: n**2 for n in numbers if n % 2 == 0}

# Print the resulting dictionary created using dictionary comprehension.
print(new_dict_comp)


# Another example using dictionary comprehension.
# Define a dictionary.
dict1 = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}

# Create a new dictionary where keys are doubled and values are doubled.
dict1_keys = {k * 2: v * 2 for (k, v) in dict1.items()}

# Print the resulting dictionary.
print(dict1_keys)


## Zip and Unzip

# Python code to demonstrate the working of zip()
# Initializing lists 
# Define a list containing names.
name = ["Manjeet", "Nikhil", "Shambhavi", "Astha"] 

# Define a list containing roll numbers.
roll_no = [4, 1, 3, 2] 

# Define a list containing marks.
marks = [40, 50, 60, 70] 

# Using zip() to map values 
mapped = zip(name, roll_no, marks) 
print(mapped)
  

# Converting values to print as set 
mapped = set(mapped) 
  
# Printing resultant values  
print("The zipped result is : ") 
print(mapped) 


# Unzipping values 
# Unzip the tuples obtained from zip() and assign them to separate lists.
namz, roll_noz, marksz = zip(*mapped) 
  
# Print a header for the unzipped result.
print("The unzipped result: \n") 

# Print the unzipped lists: name, roll number, and marks.
# Print the header for the name list.
print("The name list is : ", end = "") 
# Print the name list.
print(namz) 

# Print the header for the roll number list.
print("The roll_no list is : ", end = "") 
# Print the roll number list.
print(roll_noz)   

# Print the header for the marks list.
print("The marks list is : ", end = "") 
# Print the marks list.
print(marksz)



# Built-in Functions

### Sort

animals = ['cat', 'dog', 'sheep', 'goat', 'elephant']
# Ascending order with alphabet 
sorted(animals)


# Descending order with alphabet 
sorted(animals, reverse = True)


# Descending order with respect to length of the character
sorted(animals, key = len, reverse = True)


# Ascending order with respect to length of the character 
sorted(animals, key = len)


# Built-in function for sort
abcd = animals.sort()
print(abcd)


# Augmented Assignments with Sequences

# Explanation of mutability and immutability for list and tuple using memory location identity

# Augumented assignment operators with *=
# id of lists 
l = [1, 2, 3]
print(id(l), l)
# ID of list will not change, it will just append the values to the same ID
l *= 2 # Eql to l*2
print(id(l), l)


# Augumented assignment operators with *=
t = (1, 2, 3)
print(id(t), t)
# ID of the tuple will change instead of appending the values to the same ID 
t *= 2
print(id(t), t) 



# Handling Missing Keys

# Initializing Dictionariey
d = { 'a' : 1, 'b' : 2 }
# trying to output value of absent key
print("The Value associated with 'c' is :")
print(d['c'])

# we need to handle these kind of errors
# importing "collection" for defoault dict
import collections as cl

# Declaring defaultdict
# sets default value 'key Not found' to absent keys
defd = cl.defaultdict(lambda : 'key Not available')
# initializing values
defd['a'] = 1


# initializing values
defd['b'] = 2


# Printing value
print("The value of 'a' is : ", end = "")
print(defd['a'])


# Printing value associated with 'c'
print("The value of 'c' is : ", end = "")
print(defd['c'])



# Lambda Functions

# Defining the lambda function 
s = lambda x: x * x
s(12)


## map()
# map(function,  iterable)

# Define a list containing integers.
val = [1, 2, 3, 4, 5, 6]

# Using map() to double each element in the list and convert the result to a list.
doubled = list(map(lambda x: x * 2, val))
print(doubled)

# Using map() to convert each element in the list to a string and convert the result to a list.
converted_to_str = list(map(str, val))
print(converted_to_str)


# Using filter() to filter out odd numbers from the list and convert the result to a list.
val1 = [1, 2, 3, 4, 5, 6]
filtered_odd = list(filter(lambda x: x % 2, val1))
print(filtered_odd)

# Using filter() to filter out numbers greater than 50 from a list generated using range() and convert the result to a list.
filtered_greater_than_50 = list(filter(lambda x: x > 50, list(range(1, 100))))
print(filtered_greater_than_50)


# Using reduce() to find the product of all elements in the list.
from functools import reduce
val = [1, 2, 3, 4, 5, 6]
product = reduce(lambda x, y: x * y, val)
print(product)  # 1 * 2 * 3 * 4 * 5 * 6

# Using reduce() to find the sum of all elements in a list generated using range().
sum_of_elements = reduce(lambda x, y: x + y, list(range(1, 100)))
print(sum_of_elements)  # 1 + 2 + 3 + 4 + 5 + ........

# Print documentation for reduce() function.
help(reduce)




## Loop vs Comprehension vs Map (lambda function)

# Original list
list1 = [1, 2, 3, 4, 5]

# Using a loop to calculate squares
squares1 = []
for i in list1:
    squares1.append(i ** 2)

# Using list comprehension to calculate squares
squares2 = [i ** 2 for i in list1]

# Using map and lambda function to calculate squares
squares3 = list(map(lambda x: x ** 2, list1))

# Printing the results
squares1, squares2, squares3



## Iterators
# Iterator is an object which allows a programmer to traverse through all the 
# elements of a collection, regardless of its specific implementation.

# Define a list of names.
names = ["Rishu", 'Aayush', 'Shubh', 'Ram']

# Print the list of names.
print(names)   # iterable

# Convert the list into an iterator using the __iter__() method.
new_list = names.__iter__()
print(new_list)

# Print documentation for the iterator object.
help(new_list)

# Iterate over the iterator and print each element.
for obj in new_list:
  print(obj)

# Attempting to print an element directly from the list raises an error because lists are iterables, not iterators.
names = ["Rishu", 'Aayush', 'Shubh', 'Ram']
names   # iterable
print(next(names))  # Raises TypeError

# Attempting to print an element directly from the iterator works.
print(next(new_list))  # Prints the first element of the iterator

# Convert the list into an iterator using the iter() function.
new_list = iter(names)
print(next(new_list))  # Prints the first element of the iterator

# Print the remaining elements of the iterator using next() function.
# Attempt to retrieve the next element from the iterator 'new_list'.
print(next(new_list))  # Print the next element from the iterator.

# Attempt to retrieve the next element from the iterator 'new_list'.
print(next(new_list))  # Print the next element from the iterator.

# Attempt to retrieve the next element from the iterator 'new_list'.
print(next(new_list))  # Print the next element from the iterator.

# Iterate over the original list and print each element.
for obj in names:
  print(obj)



# Why use iterator ?

# An **iterable** is an object that has an __iter__ method which returns an # 
# iterator, or defines a __getitem__ method that can take sequential indexes 
# starting from zero. So an iterable is an object that you can get an iterator.

# An **iterator** is an object with a next (Python 2) or __next__ (Python 3) method.

# Whenever you use a 'for' loop, or map, or a list comprehension, etc., in 
# Python, the next method is called automatically to get each item from the 
# iterator, thus going through the process of iteration.

# Create a list of numbers from 0 to 999999.
names = [i for i in range(1000000)]

# Convert the list into an iterator.
new_list = iter(names)

# Import the sys module to access system-related information.
import sys

# Print the size of the 'names' variable in bytes.
print(f'size of variable names is {sys.getsizeof(names)} bytes')

# Print the size of the 'new_list' variable in bytes.
print(f'size of variable new_list is {sys.getsizeof(new_list)} bytes')

# Define an iterable string.
iterable_value = 'Machine Learning'

# Convert the string into an iterator.
iterable_obj = iter(iterable_value) 

# Iterate over the iterator until it raises a StopIteration exception.
while True: 
    try:   
        # Iterate by calling next 
        item = next(iterable_obj) 
        print(item) 
    except StopIteration:   
        # Exception will occur when iteration stops
        break


## Generator
# A generator is a function that produces or yields a sequence of values 
# using 'yield' method.

# When a generator function is called, it returns a generator object without 
# even beginning execution of the function. When the next( ) method is called
# for the first time, the function starts executing until it reaches the yield 
# statement, which returns the yielded value. The yield keeps track i.e., remembers
# the last execution and the second next( ) call continues from previous value.

# Simple function 
import sys  # Importing the sys module to access system-related information

# Define a function 'squence' that creates a list of numbers from 0 to N-1
def squence(N):
    x = []  # Initialize an empty list
    for i in range(N):  # Loop from 0 to N-1
        x.append(i)  # Append each number to the list
    return x  # Return the list

# Call the 'squence' function with argument 10000 and store the result in the 
# variable 'alist'
alist = squence(10000)

# Print the list stored in the variable 'alist'
print(alist)

# Define a generator function 'squence' that yields numbers from 0 to N-1
def squence(N):  
    for i in range(N):  # Loop from 0 to N-1
        yield i  # Yield each number

# Call the 'squence' generator function with argument 100000 and store the 
# generator object in the variable 'blist'
blist = squence(100000)

# Print the generator object stored in the variable 'blist'
print(blist)

# Print the next value from the generator 'blist'
print(next(blist))

# Print the size of the variable 'alist' in bytes
print(f'size of variable new_list is', sys.getsizeof(alist), 'bytes')

# Print the size of the variable 'blist' in bytes
print(f'size of variable new_list is', sys.getsizeof(blist), 'bytes')

# A generator function to generate Fibonacci numbers up to a specified limit 'limit'.
def fib(limit): 
      
    # Initialize first two Fibonacci Numbers  
    a, b = 0, 1  # Initialize variables 'a' and 'b' to store the first two Fibonacci numbers.

    # One by one yield next Fibonacci Number 
    while a < limit:  # Continue generating Fibonacci numbers until 'a' exceeds the specified limit.
        yield a  # Yield the current Fibonacci number.
        a, b = b, a + b  # Update 'a' and 'b' to calculate the next Fibonacci number.


# Define a generator function to generate Fibonacci numbers up to the nth term 'n'.
def fib(n):
    a, b = 0, 1  # Initialize variables 'a' and 'b' to store the first two Fibonacci numbers.
    for _ in range(n):  # Iterate 'n' times to generate 'n' Fibonacci numbers.
        yield a  # Yield the current Fibonacci number.
        a, b = b, a + b  # Update 'a' and 'b' to calculate the next Fibonacci number.


# Create a generator object by calling the fib() function with argument 5
x = fib(5) 
print(x)  # Print the generator object

# Iterating over the generator object using next() until StopIteration is raised
while True:
    try:
        print(next(x), end="\n")  # Print the next Fibonacci number
    except StopIteration:  # Catch StopIteration exception
        break  # Break out of the loop

# String slicing
String = 'DATA SCIENTIST'

# Using indexing sequence to slice the string and print substrings
print(String[:6])  # Print characters from index 0 to 5
print(String[1:5:3])  # Print characters from index 1 to 4 with step size 3
print(String[-1:-14:-3])  # Print characters from last index to 14th index (reversed) with step size 3
