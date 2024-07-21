# PYTHON PROGRAMMING

### History of Python

# - Father of Python is “Guido Van Rossum”
# - In 1989 at National Research Institute, Netherland
# - But, officially released on 20th Feb 1991

# ***Named after***
# - Monty Python's flying circus (Telecasted in BBC around 1969 - 1974)


# Python features include:

# * Functional Programming Language feature from C
# * Object Oriented Programming Language feature from C++
# * Scripting Language feature from Perl and Shell scripting
# * Modular Programming Language from Modular-3 


### Modes of Python Interpreter

# Python Interpreter reads and executes Python code. It uses 2 modes of Execution.
# - Interactive mode 
# - Script mode

# `Interactive mode`:
# Allows us to interact with OS. When a Python statement is executed, interpreter displays the result(s) immediately.

# `Script mode`:
# A Python program is written in a file and then interpreter is used to execute the content of the file
# Python scripts have the extension .py
# Scripts can be saved to disk for future use



### Python Programming Basics

# **Objects:**
# - Variables
# - Python Objects
# - Standard Types
# - Other Built-in Types
# - Internal Types
# - Standard Type Operators
# - Standard Type Built-in Functions
# - Categorizing the Standard Types
# - Unsupported Types
 

# **Numbers:**
# - Introduction to Numbers
# - Integers
# - Floating Point Real Numbers
# - Complex Numbers
# - Operators
# - Built-in Functions
# - Related Modules

# **Sequences:**
# - Strings
# - Lists
# - Tuples
# - Mapping 
# - Set Types


## Variables in Python

# - The content to be used for processing (data, functions, etc.) are stored in session memory.
# - A variable is a pointer to this content in memory. It allows us to refer to
#   a stored value by assigning a name to it.
# - Can be referred to as a Named memory locations to store values.
# - Variables can be alphanumeric but must start with an alphabet.
# - Identifiers can be a combination of letters in lowercase (a to z) or 
#   uppercase (A to Z) or digits (0 to 9) or an underscore (_).
# - No special character allowed including spaces, only ‘_’ (underscore) is allowed.
# - In Python variables are directly assigned with a value thanks to its dynamic programming.
# - Assignment operator(=) is used to assign a value on the right side and
#   variable on left side.

# ***Example***:
# - X = 10
# - x = 15.6
# - abc = ‘python’


### Keywords or Reserved Words

# - Keywords or Reserved words are pre-defined variables
# - Keywords are case sensitive
# - Cannot be used as a variable name, function name or any other identifier


import keyword  # Import the keyword module
keyword.kwlist  # Display the list of Python keywords


#  ## Variables Declaration

# - Variable assignment rules
# - Variable name should start with a letter 
# - It should contain only alphanumerics
# - It should not contain symbols except '_'
# - Should not start with a number 
# - Should not have space in between, use ( _ ) instead ex : name_last
# - Should not use any of these symbols "",<>,|,?,!,@,#,%,$,&,*,~,+
# - Should not contain predefined keywords (variables) which have special 
#   meaning in python like "list"
# - Considered best practice use lower case names as variabe assignment   


## Variable Examples

car = 10    # Assign the value 10 to the variable 'car'
print(car)  # Print the value of the variable 'car'

car 2s = 20  # This line will raise a syntax error because variable names cannot contain spaces or start with a digit.

car$ = 5  # This line will raise a syntax error because variable names cannot contain special characters except underscore.

new_car = 'car'  # Assign the string 'car' to the variable 'new_car'
print(new_car)   # Print the value of the variable 'new_car'

## Operators in Python

# Operators are the constructs which can manipulate the value of operands
# - Ex: 2 + 3 = 5
#     - where, 2 and 3 are called operands and + is called operator

# '''**Types of Operators:**
# - Arithmetic Operators
# - Comparison (Relational) Operators
# - Assignment Operators
# - Logical Operators
# - Membership Operators
# - Identity Operators
# - Bitwise Operators'''



## Arithmetic Operators

# Consider

a = 10  # Assign the value 10 to variable 'a'
b = 20  # Assign the value 20 to variable 'b'

# Addition
print(a + b)  # Output: 30 - Prints the result of adding 'a' and 'b'

# Subtraction
print(a - b)  # Output: -10 - Prints the result of subtracting 'b' from 'a'

# Multiplication
print(a * b)  # Output: 200 - Prints the result of multiplying 'a' and 'b'

# Division 
print(a / b)  # Output: 0.5 - Prints the result of dividing 'a' by 'b'

# Floor Division
print(a // b)  # Output: 0 - Prints the integer quotient of dividing 'a' by 'b'

# Modulus
print(b % a)  # Output: 0 - Prints the remainder of dividing 'b' by 'a'

# Modulus
print(a % b)  # Output: 10 - Prints the remainder of dividing 'a' by 'b'

# Exponential
print(a ** b)  # Output: 100000000000000000000 - Prints 'a' raised to the power of 'b'

x = 1; y = 2
print(x / y)  # Output: 0.5 - Prints the result of dividing 'x' by 'y'

x = 1; y = 2
print(x // y)  # Output: 0 - Prints the integer quotient of dividing 'x' by 'y'

x = -1; y = 2
print(x // y)  # Output: -1 - Prints the integer quotient of dividing 'x' by 'y'


## Comparision Operator
- Returns bool values

print(a == b)  # Output: False - Checks if 'a' is equal to 'b'
 
print(a != b)  # Output: True - Checks if 'a' is not equal to 'b'
 
print(a > b)   # Output: False - Checks if 'a' is greater than 'b'
 
print(a < b)   # Output: True - Checks if 'a' is less than 'b'

print(a >= b)  # Output: False - Checks if 'a' is greater than or equal to 'b'

print(a <= b)  # Output: True - Checks if 'a' is less than or equal to 'b'


## Assignment Operators

c = a + b  
print(c)      # Output: 30 - Adds 'a' and 'b', assigns result to 'c'

c += b        # Can be read as c = c + b
print(c)      # Output: 50 - Adds 'b' to 'c' and updates 'c'

c -= b
print(c)      # Output: 30 - Subtracts 'b' from 'c' and updates 'c'

c *= b
print(c)      # Output: 600 - Multiplies 'c' by 'b' and updates 'c'

c /= b
print(c)      # Output: 30.0 - Divides 'c' by 'b' and updates 'c'

c %= b
print(c)      # Output: 10.0 - Computes modulus of 'c' with 'b' and updates 'c'

c **= b
print(c)      # Output: 100000000000000000000 - Raises 'c' to the power of 'b' and updates 'c'

c //= b
print(c)      # Output: 5000000000000000000 - Computes floor division of 'c' by 'b' and updates 'c'


## Membership Operators
# - It will check whether left value is a member of right value or not 

print("y" in "Python")   # Output: True - Checks if 'y' is in the string "Python"

print("l" in "Python")   # Output: True - Checks if 'l' is in the string "Python"

print("p" in "Python")   # Output: False - Checks if 'p' is in the string "Python"

print("P" not in "Python")  # Output: True - Checks if 'P' is not in the string "Python"


## Identity Operators
# - It will check whether left value is equal to the right value or not

"y" is "Python"  # Output: False - Checks if the string "y" is the same object as the string "Python"

print(1 is 1)    # Output: True - Checks if the integer 1 is the same object as the integer 1

print(2 is 1)    # Output: False - Checks if the integer 2 is the same object as the integer 1

print("python" is "Python")  # Output: False - Checks if the string "python" is the same object as the string "Python"

print(1 is 0)    # Output: False - Checks if the integer 1 is the same object as the integer 0

print(1 is not 1)  # Output: False - Checks if the integer 1 is not the same object as the integer 1

print("hi hello"  is not  "hello hi")  # Output: True - Checks if the string "hi hello" is not the same object as the string "hello hi"

## Precedence of Operators
# - If an expression contains more than one operator, then order of evaluation 
#   depends on the order of operations
# - `PEMDAS` (Parentheses, Exponentiation, Multiplication, Division, Addition, Subtraction)
#     - Parentheses have the highest precedence and can be used to force an 
#       expression to evaluate in the order that we want
#     - Exponentiation has the next highest precedence
#     - Multiplication and Division have higher precedence than Addition and Subtraction
#     - Operators with the same precedence are evaluated from left to right 
#       (except exponentiation)



## Standard Types

### Python Data Types

# - Every value in Python has a data type. It is a set of values, and the 
#   allowable operations on those values.
# - Data types are an essential concept in any computer programming language.
# - It helps to understand the kind of operations that can be performed.
# - Data Types are of 5 types:
#     - Numbers
#     - Boolean
#     - Sets 
#     - Mapping (Dictionaries)
#     - Sequences


### Numerical Types

# - Integer         ---> int     ---> Whole numbers such as 3, 300, 200
# - Floating Point  ---> float   ---> Numbers with decimal point 3.56, 2.45, 76.87
# - Complex Numbers ---> complex ---> Numbers with real and imaginary parts 2+3j
# - Boolean         ---> bool    ---> True or False (True = 1 and False = 0)


### Scalar Data Types

# **Numbers:**
# Number data type stores numerical values.
# - Integers: Positive or Negative whole numbers (non-decimals) [ex: 20]
# - Floating point numbers: Decimal values [ex: 20.14]
# - Complex numbers: Represented in a + bj (‘a’ is real number and ‘b’ is imaginary part) 


# **Boolean:**
# - Boolean type has only two possible values:
#     - True
#     - False
# - Bool is used to test whether the result of an expression is true or false
# - True = 1
# - False = 0


# Integer
print(1)          # Output: 1
print(2 + 4)      # Output: 6 (Addition)
print(12 * 3)     # Output: 36 (Multiplication)
print(12 / 3)     # Output: 4.0 (Division)

# Float
print('Float')
print(11.5)       # Output: 11.5
x = 55.45
print(x)          # Output: 55.45
print(type(x))    # Output: <class 'float'>

# Complex number
print('Complex Number')
print(2 + 3j)     # Output: (2+3j)
type(2 + 3j)       # Output: <class 'complex'>

# Boolean
# bool --> Logical values indicating True or False
print(type(True))   # Output: <class 'bool'>
print(type(False))  # Output: <class 'bool'>




# Sequence Types
# - String ---> str ---> Ordered sequence of characters
# - Lists ---> list ---> Ordered sequence of objects
# - Tuples ---> tup ---> Ordered immutable sequence of objects


### Python Data Types: Strings

# **Strings:**
# - A series or sequence of characters - letters, numbers, and special characters.
# - They are marked by quotes: single (‘ ’) or double (" ") or triple (‘’’ ‘’’), (""" """).
# - Characters can be accessed using indexing and slicing operations.
# - Strings are immutable i.e.; the contents of a string cannot be changed once created.
# - Positive indexing starts from 0, and Negative subscript helps in accessing 
#   the string from the end which starts from -1.


### Strings

# Strings are ordered sequence, that means we can use indexing and slicing to 
# grab sub-sections of the string.

# Indexing notation uses square brakets [ ].

# A number to indicate positon of the value we wish to access can be mentioned 
# within square brakets [ ].

a = """ This data has three types of flower classes: 
    Setosa, Versicolour, and Virginica. The dataset is available in the 
    scikit-learn library, or you can also download it from the UCI Machine 
    Learning Library."""

print(a,'\n', a)  # Output: Prints the value of 'a' (multi-line string)
type(a)         # Output: <class 'str'> (prints the data type of 'a')
print('Datatype is', type(a),'\n')  # Output: Prints the data type of 'a'

s1 = 'sharat is trainer'

print('Length is', len(s1))  # Output: Length is 17 (prints the length of string 's1')

### Syntax of indexing of strings

# - String_name[index]
# - Positive index starts from 0
# - Negative index starts from -1 (Reverse order)


### Slicing of strings

# [start : stop : step] this is called as slicing of strings.

# - Start is a numerical index for slice start.
# - Stop is the index you go upto (but not included).
# - Step is the jump you take.

# Accessing values in string 
Name = "Digitmg"

print('Indexing of strings')
print(Name[0])  # Positive Indexing: Prints the first character of the string
print(Name[3])  # Positive Indexing: Prints the fourth character of the string

print(Name[-1]) # Negative Indexing: Prints the last character of the string

print('\nSlicing of strings')
print(Name[1:4])  # Slicing: Prints characters from index 1 to index 3 (excluding index 4)
print(Name[-3:-1])  # Slicing: Prints characters from index -3 to index -2 (excluding index -1)

print(len(Name))  # Output: 7 (prints the length of the string 'Name')


# Update strings
var1 = 'Hello World!  '

var1
var1 + 'Python'  # Concatenation: Appends 'Python' to the end of var1

print('Hello')  # Output: Hello
print("Updated String :- ", var1 + 'Python')  # Output: Updated String :-  Hello World!  Python
print("Updated String :- ", var1[:6] + 'Python')  # Output: Updated String :-  Hello Python

print("hello \nworld")  # Output: hello 
                         #         world (prints 'hello' in one line and 'world' in the next line)

# String Formatting

format = '%.3f %s is $%d'  # Defines a format string
print(format %(6.4560, 'python prasad', 1))  # Output: 6.456 python prasad is $1


# String Formatting 

print("My name is %s and weight is %d kgs!" % ('Divit', 20.5))  # Output: My name is Divit and weight is 20 kgs!


# Example:
Name = input("Enter your name: ")  # Asks for user input for their name
type(Name)  # Output: str (prints the data type of the input)

Weight = int(input("Enter your Weight: "))  # Asks for user input for their weight and converts it to an integer
type(Weight)  # Output: int (prints the data type of the input)


print("My name is %s and my Weight is %d" %(Name, Weight))

print('This is a string {}' . format('inserted'))

print("My name is {} and my Weight is {}" . format(Name, Weight))

print("My name is {1} and my Weight is {0}" . format(Name, Weight))

print('the output is {r}' . format(r = Weight))


print(f'{Name} is {Weight} kgs') # fstrings

type(Weight)


# Triple Quotes

Statement = """my name is "Bahubali" and my age is "25".I stay in Maahishmathi"""

type(Statement)

Name = 'Sharat'
# To make string at center in given spaces
Name.center(30)
len(Name.center(50))

# count() method returns the number of occurrences of the substring in the given string.
string = "Yashvi is a trainer"

substring = "i"

count = string.count(substring)

print("The count is:", count)

# Count number of occurrences of a given substring using start and end
# Define string
string = "Sharat is a trainer"
substring = "i"

len(string)

# Count after first 'i' and before the last 'i'
count = string.count(substring, 7, 19)

# Print count
print("The count is:", count)

# Returns true if string has atleast 1 character and all characters are 
# alphanumeric; false otherwise.

Num = 'thisis34'  # No space in this string
print(Num.isalnum())

Num = "this is string examplehi 123"
Num.isalnum()

# This method returns true if all characters in the string are alphabetic and 
# there is atleast one character; false otherwise.
Num = "thisis" # No space & digit in this string
Num.isalpha()

Num = "this is string example0909090!!!"
Num.isalpha()

# This method returns true if all characters in the string are digits and there
# is at least one character, false otherwise.

Num = "123456";  # Only digit in this string
Num.isdigit()  # This will raise an error because 'Num' is not defined before this line.

Num = "this is string example!!!"
Num.isdigit()  # Output: False, as the string contains non-digit characters

Name = "YASHvi"
type(Name)  # Output: <class 'str'>, indicating that 'Name' is a string

print(Name)  # Output: YASHvi

# To make the first letter capital
print(Name.capitalize())  # Output: Yashvi

print(Name.upper())  # Output: YASHVI
print(Name.lower())  # Output: yashvi
print(Name.swapcase())  # Output: yashVI


# This method returns a copy of the string in which all case-based characters 
# have been lowercased.
Num = "THIS IS STRING EXAMPLE!!!"
Num.lower()  # Output: 'this is string example!!!'

# This method returns a copy of the string in which all case-based characters 
# have been lowercased.

Num = "this is string example!!!"
Num.upper()  # Output: 'THIS IS STRING EXAMPLE!!!'

# The following example shows the usage of the replace() method.

reply = "this is a string example!!! Is this really is a string"

print(reply.replace("is", "was"))
# Output: 'thwas was a string example!!! Is thwas really was a string'

print(reply.replace("is", "was", 2))
# Output: 'thwas was a string example!!! Is this really is a string'

print(":".join(reply))
# Output: 't:h:i:s: :i:s: :a: :s:t:r:i:n:g: :e:x:a:m:p:l:e:!:!:!: :I:s: :t:h:i:s: :r:e:a:l:l:y: :i:s: :a: :s:t:r:i:n:g'

# The following example shows the usage of the split() method.

split1 = "Line1-abcdef Line2-abc \nLine4-abcd"
print(split1)
# Output: Line1-abcdef Line2-abc 
# Line4-abcd

print(split1.split())
# Output: ['Line1-abcdef', 'Line2-abc', 'Line4-abcd']

print(split1.split(' ', 1 ))
# Output: ['Line1-abcdef', 'Line2-abc \nLine4-abcd']



### Python Data Types: Lists

# **Lists:**
# - A List is an ordered sequence of items
# - Represented using square brackets [ ]
# - Values in the list are called elements/items
# - It can be written as a list of comma-separated items (values)
# - Items in the list can be of different data types
# - Values of the list are mutable (values can be changed)

list1 = ['Sharat', '360DigiTMG', 2013, 2018]
list1 = ['ssss', 5, 5.5, 40+55j, True]
print(list1)  # Output: ['ssss', 5, 5.5, (40+55j), True]

# Access values in the variable using index numbers
print(list1[0])  # Output: 'ssss'

print(list1[3])  # Output: (40+55j)

print(list1[1:4])  # Output: [5, 5.5, (40+55j)]

# Append: add new elements to the existing list
aList = [123, 'xyz', 'zara', 'abc']

aList[2] = 55 + 40j

print(aList)  # Output: [123, 'xyz', (55+40j), 'abc']

aList.append(2009)
print(aList)  # Output: [123, 'xyz', (55+40j), 'abc', 2009]

# Pop: remove the element from the existing list
print(aList.pop())  # Output: 2009
print(aList)  # Output: [123, 'xyz', (55+40j), 'abc']


# Pop the element using index number
print(aList.pop(2))  # Output: 'tommy'
print(aList)  # Output: [123, 'xyz', 'abc']

# Insert: insert a value using index number 
aList = [123, 'xyz', 'tommy', 'abc', 123]

aList.insert(3, 2009)
print(aList)  # Output: [123, 'xyz', 'tommy', 2009, 'abc', 123]


# Append vs Extend  

# Append
aList = [123, 'xyz', 'tommy', 'abc', 123]
bList = [2009, 'beneli']

aList.append(bList)
print(aList)  # Output: [123, 'xyz', 'tommy', 'abc', 123, [2009, 'beneli']]

# Extend
aList = [123, 'xyz', 'tommy', 'abc', 123]
bList = [2009, 'beneli']

aList.extend(bList)
print(aList)  # Output: [123, 'xyz', 'tommy', 'abc', 123, 2009, 'beneli']

bList.append(aList)  # Appending the entire aList as an element
print(bList)  # Output: [2009, 'beneli', [123, 'xyz', 'tommy', 'abc', 123, 2009, 'beneli']]

bList.extend(aList)  # Extending bList with elements of aList
print(bList)  # Output: [2009, 'beneli', [123, 'xyz', 'tommy', 'abc', 123, 2009, 'beneli'], 123, 'xyz', 'tommy', 'abc', 123, 2009, 'beneli']


# Reverse: to reverse the given list 
aList = [123, 'xyz', 'tommy', 'abc', 123]
aList.reverse()
print(aList)  # Output: [123, 'abc', 'tommy', 'xyz', 123]

# Sort: sort the given list from ascending or descending
blist = [8, 99, 45, 33]
blist.sort()
print(blist)  # Output: [8, 33, 45, 99]

# count: count the value in given list of elements
aList = [123, 'xyz', 'zara', 'abc', 123, "zara"]
print(aList.count(123))  # Output: 2

# Index: find the index of the first occurrence of a value
print(aList.index('zara'))  # Output: 2


# Arithmetic operations on Lists
list1 = [1, 2, 3]
list2 = [1, 2, 3]

print(1 + 2)  # Output: 3
print('a' + 'b')  # Output: 'ab'

list3 = list1 + list2
print(list3)  # Output: [1, 2, 3, 1, 2, 3]

print(list1 * 3)  # Output: [1, 2, 3, 1, 2, 3, 1, 2, 3]

# Display the functions for the list data type
dir(list)




### Python Data Types: Tuples

# **Tuples:**
# - A tuple is the same as a list, with some difference
# - Tuple values are enclosed in parentheses ()
# - A tuple is an immutable list i.e., once a tuple is created, it cannot be edited
# - Benefits of Tuple: 
#     - Tuples are faster than lists
#     - If the user wants to protect the data from accidental changes, a tuple can be used
#     - Tuples can be used as keys in dictionaries, while lists can't


# Create a tuple dataset

tup1 = ('sharat', '360digitmg', 'Fiat', 8055)
tup2 = (1, 2, 3, 4, 5 )

# Create a empty tuple
tup1 = ()
tup1

# Create a single tuple
tup1 = (50,)

type(tup1)

# Accessing Values in Tuples
tup1 = ('sharat', '360digitmg', 'Fiat', 8055, 2 + 3j, False)
print(tup1[0])

tup2 = (11, 12, 13, 14, 15, 16, 17)
print(tup2[1:5])

# Updating Tuples
tup1 = (12, 34.56)
tup2 = ('abc', 'xyz')

# Tuples are immutable
tup1[1] = 40 # this line will throw error. Tuple is immutable, hence values cannot be changed. 

s = 'sam'
s[0]

s[0] = 'z' # this line will throw error. Tuple is immutable, hence values cannot be changed. 

# Create a new tuple

tup3 = tup1 + tup2
print(tup3)

# index, count
tup = (1, 2, 3, 4, 1, 2)

# Note: 'tup' is not defined in the provided code. You need to define 'tup' as 
# a tuple before using these operations.


# Example:
tup = (1, 2, 3, 4, 5, 2, 2)
print(tup.index(1))  # Output: 0 (index of the first occurrence of 1)

print(tup.count(2))  # Output: 3 (count of occurrences of 2)

# Delete Tuple Elements
tup = ('sharat', '360digitmg', 'Fiat', 8055)
print(tup)

# Delete the given tuple 
del(tup)
tup

help(tuple)

dir(tuple)



### Python Data Types: Sets

# **Sets:**
# - Sets are used to store multiple items in a single variable.
# - Sets are unindexed
# - Sets are unordered
# - Sets can be altered
# - Sets do not allow duplicates
# - Frozen sets are unchangeable


# Sets --> set ---> Unordered collection of unique objects

# A normal Set
# Define a set containing integers 1, 2, and 3
normal_set = {1, 2, 3}
print(normal_set)  # Print the set containing integers 1, 2, and 3

# Define a set using the set() constructor with a list of strings as input
normal_set = set(["a", "b", "c", "f"])
print(normal_set)  # Print the set containing strings 'a', 'b', 'c', and 'f'

# Define a set directly with curly braces containing strings 'a', 'b', 'c', and 'd'
normal_set = {'a', 'b', 'c', 'd'}
print(normal_set)  # Print the set containing strings 'a', 'b', 'c', and 'd'


# Add a duplicate value to the 'set'. No duplicates are allowed in sets, hence you will not see the duplicate entry
# Add element "d" to the set
normal_set.add("d")
print(normal_set)  # Print the updated set

# Converting a list into a set
my_list = [1, 1, 2, 3, 4, 4, 5, 5, 6, 7]
my_set = set(my_list)  # Convert the list into a set to remove duplicates
print(my_set)  # Print the resulting set without duplicates

# Operations on sets
set1 = {1, 2, 3}
set2 = {1, 9, 0}

# Find the intersection of set1 and set2
intersection_set = set1.intersection(set2)
print(intersection_set)  # Print the intersection set

# Convert the intersection set to a list
intersection_as_list = list(intersection_set)
print(intersection_as_list)  # Print the intersection as a list


# intersection function can be represented as (&) 
set1 = {1, 2, 3}
set2 = {1, 9, 0}
set3 = set1 & set2 # set1.intersection(set2)

# Union of two sets
print("Intersection of set1 & set2 is: set3 =", set3)

# union() function

set1 = {2, 4, 5, 6}
set2 = {4, 6, 7, 8}
set3 = {7, 8, 9, 10}

# Union of two sets
print("set1 U set2 : ", set1.union(set2))

# union() function can be represented as vertical slash (|) which is pipe delimiter

set1 = {2, 4, 5, 6}
set2 = {4, 6, 7, 8}

set3 = set1 | set2 #set1.union(set2)

# Union of two sets
# Printing the union of set1 and set2
print("Union of set1 & set2 is: set3 =", set3)

# Defining three sets
set1 = {2, 4, 5, 6}
set2 = {4, 6, 7, 8}
set3 = {7, 8, 9, 10}

# Printing the union of set1, set2, and set3
print("set1 U set2 U set3 :", set1.union(set2, set3))

# Defining two sets
set1 = {1,2,3,4,5,6,7,8,9}
set2 = {3,4,5,6,7,8,9}

# Calculating the union of set1 and set2 using '|' operator
set3 = set1 | set2

# Calculating the intersection of set1 and set2 using '&' operator
set4 = set1 & set2

# Printing the union of set1 and set2
print('intersection of set1 | set2 is: set3 =', set3)

# Printing the intersection of set1 and set2
print('intersection of set1 & set2 is: set4 =', set4)


# difference between set3 and set4
set5 = set3 - set4
print("Elements in set3 and not in set4: set5 = ", set5)
print("\n")

# check if set4 and set5 are disjoint sets
if set4.isdisjoint(set5):
    print("set4 and set5 have nothing in common\n")

# Removing all the values of set5
set5.clear()     # set is a mutable object

print("After applying clear on sets set5: ")
print("set5 = ", set5)


### Frozen set 
# - Frozen set is just an immutable version of a Python set object.
# - While elements of a 'set' can be modified at any time, elements of the 
#   'frozen set' remains the same after creation.
# - Syntax: frozenset([iterable])


# Define a tuple of vowels
vowels = ('a', 'e', 'i', 'o', 'u')

# Create a frozen set from the tuple
fSet = frozenset(vowels)

# Print the frozen set
print('The frozen set is:', fSet)

# Create an empty frozen set
empty_fSet = frozenset()

# Print the empty frozen set
print('The empty frozen set is:', empty_fSet)


# frozensets are immutable
fSet.add('v')

# To get detailed info about set
help(set)



# Mappings / Dictionaries

# **Mappings / Dictionaries:**

# - Dictionaries are unordered Key-value-pairs.
# - Keys are Immutable, where as, the values are mutable. 
# - Dictionary is created by using curly brackets - {}.
# - Dictionaries are accessed via keys and not via their position. 
# - A dictionary is an associative array (also known as hashes). Any key of the 
#   dictionary is associated (or mapped) to a value(s). 
# - The values of a dictionary can be any Python data type.
# - Dictionaries don't support the sequence operation of the sequence data 
#   types like strings, tuples and lists.

# Dictionaries ---> dict ---> Unordered key value pairs {'mykey':'value'} 
# (Also known as Mapping Type).

# Accessing values in dictionary

# Define a dictionary
dict1 = {'Name': 'Sharat', 'Age': 40, 'bike': 'Honda'}

# Print the dictionary
print(dict1)

# Access value using key
print(dict1['Name'])
print(dict1['Age'])

# Updating dictionary
dict1 = {'Name': 'Divit', 'Age': 8, 'bike': 'Bicycle'}

# Update existing entry
dict1['Age'] = 9

# Add new entry
dict1['School'] = "DPS School"

dict1['Salary'] = 50000

# Print updated values
print(dict1['Age'])
print(dict1['School'])
print(dict1['Salary'])

# Retrieve keys, values, and items of the dictionary
print(dict1.keys())
print(dict1.values())
print(dict1.items())


# Delete dictionary elements
dict1 = {'Name': 'Sharat', 'Age': 40, 'bike': 'Honda'}

# Remove entry with key 'Name'
del(dict1['Name'])
dict1

# Remove all entries in given dictionary
dict1.clear()
dict1

# Delete entire dictionary
del(dict1)

# Once dictionary is deleted the object does not exist
print(dict1['Age'])

# Accessing elements in a dictionary
d = {'k1':123,'k2':[8,9,10],'k3':{'inside_key':100}}

# Print the dictionary
print(d)

# Access value using key
print(d['k1'])

# Access value of a list within the dictionary
print(d['k2'])

# Access specific element within the list
print(d['k2'][1])

# Access value of a nested dictionary within the dictionary
print(d['k3'])

# Access value of a key within the nested dictionary
print(d['k3']['inside_key'])




## Dates and Times
# To deal with dates and times we have seperate packages in Python - datetime
from datetime import date

# Creating a date object from ISO format string
date_obj = date.fromisoformat('2019-12-04')

# Printing the date object
print(date_obj)  # Output: 2019-12-04

# Creating a bytes object with length 3
byt = bytes(3)

# Printing the type of the object
print(type(byt))  # Output: <class 'bytes'>

# Printing the bytes object
print(byt)  # Output: b'\x00\x00\x00'


# We can replace the dates
d = date(2002, 12, 31)
d.replace(day=26)

# Example
# date.isocalendar()
from datetime import date
date(2003, 12, 29).isocalendar()

# date.isoformat()
from datetime import date
date(2002, 12, 4).isoformat()


# Timedelta function demonstration    
from datetime import datetime, timedelta 
  
  
# Using current time 
ini_time_for_now = datetime.now() 
  
# printing initial_date 
print ("initial_date", str(ini_time_for_now)) 
  
# Calculating future dates 
# for two years 
future_date_after_2yrs = ini_time_for_now + timedelta(days = 730)
# printing calculated future_dates 
print('future_date_after_2yrs:', str(future_date_after_2yrs))

import time
date.fromtimestamp(time.time())

"""## **Date Formatting**

%a	Abbreviated weekday name.	Sun, Mon, ...

%A	Full weekday name.	Sunday, Monday, ...

%w	Weekday as a decimal number.	0, 1, ..., 6

%d	Day of the month as a zero-padded decimal.	01, 02, ..., 31

%-d	Day of the month as a decimal number.	1, 2, ..., 30

%b	Abbreviated month name.	Jan, Feb, ..., Dec

%B	Full month name.	January, February, ...

%m	Month as a zero-padded decimal number.	01, 02, ..., 12

%-m	Month as a decimal number.	1, 2, ..., 12

%y	Year without century as a zero-padded decimal number.	00, 01, ..., 99

%-y	Year without century as a decimal number.	0, 1, ..., 99

%Y	Year with century as a decimal number.	2013, 2019 etc.

%H	Hour (24-hour clock) as a zero-padded decimal number.	00, 01, ..., 23

%-H	Hour (24-hour clock) as a decimal number.	0, 1, ..., 23

%I	Hour (12-hour clock) as a zero-padded decimal number.	01, 02, ..., 12

%-I	Hour (12-hour clock) as a decimal number.	1, 2, ... 12

%p	Locale’s AM or PM.	AM, PM

%M	Minute as a zero-padded decimal number.	00, 01, ..., 59

%-M	Minute as a decimal number.	0, 1, ..., 59

%S	Second as a zero-padded decimal number.	00, 01, ..., 59

%-S	Second as a decimal number.	0, 1, ..., 59

%f	Microsecond as a decimal number, zero-padded on the left.	000000 - 999999

%z	UTC offset in the form +HHMM or -HHMM.	 

%Z	Time zone name.	 

%j	Day of the year as a zero-padded decimal number.	001, 002, ..., 366

%-j	Day of the year as a decimal number.	1, 2, ..., 366

%U	Week number of the year (Sunday as the first day of the week). All days in a new year preceding the first Sunday are considered to be in week 0.	00, 01, ..., 53

%W	Week number of the year (Monday as the first day of the week). All days in a new year preceding the first Monday are considered to be in week 0.	00, 01, ..., 53

%c	Locale’s appropriate date and time representation.	Mon Sep 30 07:06:05 2013

%x	Locale’s appropriate date representation.	09/30/13

%X	Locale’s appropriate time representation.	07:06:05

%%	A literal '%' character.
"""


#import datetime module
import datetime

# Define the date-time string
date_time_str = '2018-06-29 08:15:27.243860'

# Convert the string to a datetime object
date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')

# Print the date, time, and date-time
print('Date:', date_time_obj.date())
print('Time:', date_time_obj.time())
print('Date-time:', date_time_obj)

# Define another date-time string with a different format
date_time_str = '2018-jan-29 08:15:27'

# Convert the string to a datetime object
date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%b-%d %H:%M:%S')

# Print the datetime object
print(date_time_obj)  # Output: 2018-01-29 08:15:27

# Formatting examples
print(date_time_obj.strftime('%Y-%b-%d'))  # Output: 2018-Jan-29
print(date_time_obj.strftime('%a-%b-%y'))  # Output: Mon-Jan-18
print(date_time_obj.strftime('%a-%b-%y %I:%M:%S %p'))  # Output: Mon-Jan-18 08:15:27 AM
print(date_time_obj.strftime('Date is %D, and Time is %T'))  # Output: Date is 01/29/18, and Time is 08:15:27
print(date_time_obj.strftime('%A-%B-%Y'))  # Output: Monday-January-2018



## Categorizing the Standard Types

# - Data Type --------> Storage Model ---> Update Model ---> Access Model
# - Numbers ----------> Scalar ----------> Immutable ------> Direct
# - Strings ----------> Scalar ----------> Immutable ------> Sequence
# - Lists ------------> Container -------> Mutable --------> Sequence
# - Tuples -----------> Container -------> Immutable ------> Sequence
# - Dictionaries -----> Container -------> Mutable --------> Mapping
# - Datetime

