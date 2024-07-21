# Conditional Statements / Control Flow

# Example: - if condition 

a = 23  # Assigns the value 23 to variable 'a'.

if a >= 22:  # Checks if the value of 'a' is greater than or equal to 22.
   print("if")  # Prints "if" if the condition is true.



x = 3  # Assigns the value 3 to variable 'x'.

if x < 5:  # Checks if the value of 'x' is less than 5.
    print(x)  # Prints the value of 'x' if the condition is true.
print(5)  # Prints the number 5 regardless of the condition.

x = 2  # Reassigns the value 2 to variable 'x'.
y = 25  # Assigns the value 25 to variable 'y'.

print(x)  # Prints the value of 'x' which is 2.

if x < 10:  # Checks if the value of 'x' is less than 10.
    print(x)  # Prints the value of 'x' if the condition is true.
print(y)  # Prints the value of 'y' which is 25 regardless of the condition.


# Example: - if and else conditions when you have two conditions

is_male = False  # Assigns the boolean value False to the variable 'is_male'.

if is_male:  # Checks if 'is_male' evaluates to True.
    print("you are male")  # Prints "you are male" if 'is_male' is True.
else:
    print("you are female")  # Prints "you are female" if 'is_male' is False.

a = 23  # Assigns the value 23 to variable 'a'.

if a >= 22:  # Checks if the value of 'a' is greater than or equal to 22.
   print("if")  # Prints "if" if the condition is true.
   print("one more statement")  # Prints "one more statement" if the condition is true.
elif a >= 21:  # Checks if the previous condition is not met and if the value of 'a' is greater than or equal to 21.
   print("elif")  # Prints "elif" if the condition is true.
else:
   print("else")  # Prints "else" if none of the above conditions are true.


# Example: - if you have more than two conditions use elif 

is_male = True  # Assigns the boolean value True to the variable 'is_male'.
is_tall = True  # Assigns the boolean value True to the variable 'is_tall'.

if is_male and is_tall:  # Checks if both 'is_male' and 'is_tall' are True.
    print("you are tall male")  # Prints "you are tall male" if both conditions are True.
elif is_male and not(is_tall):  # Checks if 'is_male' is True and 'is_tall' is False.
    print("you are short male")  # Prints "you are short male" if 'is_male' is True and 'is_tall' is False.
elif not(is_male) and is_tall:  # Checks if 'is_male' is False and 'is_tall' is True.
    print("you are not male and tall")  # Prints "you are not male and tall" if 'is_male' is False and 'is_tall' is True.
else:  # If none of the above conditions are True.
    print("you are not male and not tall")  # Prints "you are not male and not tall".



### Nested if Statement
#Example: - conditional statement inside a conditional statement


score = 500  # Assigns the value 500 to the variable 'score'.
money = 6000  # Assigns the value 6000 to the variable 'money'.
age = 65  # Assigns the value 65 to the variable 'age'.

if score > 100:  # Checks if the value of 'score' is greater than 100.
    print("You got good points")  # Prints "You got good points" if the condition is true.
    
    if money >= 5000:  # Checks if the value of 'money' is greater than or equal to 5000.
        print("You win")  # Prints "You win" if the condition is true.
        
        if age >= 30:  # Checks if the value of 'age' is greater than or equal to 30.
            print("You win in middle age")  # Prints "You win in middle age" if the condition is true.
        else:
            print("You win in young age")  # Prints "You win in young age" if the condition is false.
    else:
        print("You have high points but you do not have enough money")  # Prints message if 'money' is less than 5000.
else:
    print("You are loser")  # Prints "You are loser" if the condition for 'score' is false.


# **Loops in Python**

# **Why Loops ?**

# We use loops because if we need to repeat some set of instructions multiple
# times then instead of writing multiple lines fo code we can use loops to 
# perform those operation in a few line and with easy process. 
# Let's go through an example - if you have to print "hello world" 1000 times 
# then we need to write print statement 1000 time but using loops you can 
# easily do it in a few lines.


# **Loops :** Loops in programming language means repeating a set of 
# instructions until a specific condition is fulfilled.

# **Types of loops in Python ?**


# 1. For Loop: Repeats the set of instructions until condition is false. 
#              It checks condition before excuting the body part of loop 
# 2. While Loop: Executes set of instructions and the code that manage the loop
# 3. Nested loop: Here you can use loop inside the 'for' loops or 'while' loops


# **How to control loops or control statements in Python ?**

# * Continue statement: Continue specific part of loop till the condition 
#                       is true and leave remaining part of the loop
# * Break statement: Break the loop and get out of loop when condition is met
# * Pass Statement: When we don't want to run the condition or do not want 
#                   to print anything then we use pass statement


# # For loop
# To use 'for' loop we use **for** keyword with a variable or range, or 
# pre-defined data types (i.e., set, tuple, list, dict, string). This loop ends with a colon.

# Next line always starts with indentation (four space).

# If we don't use indentation then we can get out of the loop


# Example for 'for loop' 

snacks = ['pizza', 'burger', 'shawarma', 'franky']  # Defines a list of snacks.

for snack in snacks:  # Iterates over each element in the list 'snacks'.
    print("current snack: ", snack)  # Prints the current snack during each iteration.
print("Good day!")  # Prints "Good day!" after the loop completes.

range(0, 5)  # Declares a range object from 0 to 4, but the output is not utilized or printed.

a = list(range(0, 15))  # Attempts to call a list but encounters a TypeError due to using 'list' as a variable name.

range(1, 5, 1)  # Declares a range object from 1 to 4 with a step size of 1.

for i in range(1, 6, 2):  # Iterates over values from 1 to 5 with a step size of 2.
    print(i)  # Prints the current value of 'i' during each iteration.
else:  # Executes when the loop completes without encountering a 'break' statement.
    print("The for loop is over")  # Prints a message indicating the loop is over.

for i in range(1, 15):  # Iterates over values from 1 to 14.
    print("20 hours class of Day ", i)  # Prints a message indicating the class day during each iteration.


# Getting even number without any condition

for i in range(0, 15, 2):
  print('Number is even', i)

for i in range(1, 15, 2):
  print('Number is odd', i)
  
  
# Traversing a list

numbers = [2, 3, 5]  # Defines a list of numbers.

sum = 2  # Initializes a variable 'sum' to 0.

for num in numbers:  # Iterates over each element in the list 'numbers'.
    sum += num  # Adds the current number to the sum.
print(sum)  # Prints the total sum of the numbers in the list.

numbers = [2, 3, 5]  # Reassigns the list 'numbers'.

getsum = [i + 2 for i in numbers]  # Generates a new list by adding 2 to each element of 'numbers'.
print(getsum)  # Prints the modified list 'getsum'.

numbers = [2, 3, 5]  # Reassigns the list 'numbers'.

getnum = [i + 2 for i in numbers if i < 5]  # Generates a new list by adding 2 to elements of 'numbers' less than 5.
print(getnum)  # Prints the modified list 'getnum'.

# Example:
num = int(input("number: "))  # Takes an integer input from the user.

factorial = 1  # Initializes the factorial variable to 1.

if num < 0:  # Checks if the input number is negative.
    print("must be positive")  # Prints a message indicating that the number must be positive.
elif num == 0:  # Checks if the input number is 0.
    print("factorial = 1")  # Prints a message indicating the factorial of 0 is 1.
else:  # Executes if the input number is positive.
    for i in range(1, num + 1):  # Iterates from 1 to the input number.
        factorial = factorial * i  # Calculates the factorial.
    print("factorial =", factorial)  # Prints the calculated factorial.


## Pass
names = ["Rishu", 'Aayush', 'Ram', 'Shubh']  # Defines a list of names.

for i in names:  # Iterates through each name in the list.
    pass  # Placeholder statement indicating no action to be performed.


## Continue
names = ["Rishu", 'Aayush', 'Ram', 'Shubh']  # Defines a list of names.

for i in names:  # Iterates through each name in the list.
  if len(i) == 6:  # Checks if the length of the name is 3 characters.
    continue  # Skips to the next iteration if the length is 3.
  print(f'My name is {i}')  # Prints the name if its length is not 3.


## Break
names = ["Rishu", 'Aayush', 'Ram', 'Shubh']  # Defines a list of names.

for i in names:  # Iterates through each name in the list.
  if len(i) == 4:  # Checks if the length of the name is 3 characters.
    break  # Exits the loop if the length is 3.
  print(f'My name is {i}')  # Prints the name if its length is not 3.

for i in range(1, 10):  # Iterates from 1 to 9.
       if i == 5:  # Checks if the current value is 5.
           break  # Exits the loop if the current value is 5.
       print(i)  # Prints the current value.
print('Done')  # Prints 'Done' after the loop completes.


## For - else
# If the 'else' statement is used with a 'for' loop, the 'else' block is 
# executed only if 'for' loops terminates normally (and not by encountering 
# break statement)

# Example
numbers = [11, 33, 55, 39, 55, 75, 37, 21, 23, 41, 13]

# Example of using 'for' loop with 'else' statement
# In this loop, it iterates through each number in the list 'numbers'.
# If it encounters an even number, it prints a message and breaks out of the loop.
# If the loop completes without encountering any even number, the 'else' block is executed.
numbers = [11, 33, 55, 39, 55, 75, 37, 21, 23, 41, 13]
for num in numbers:
   if num % 2 == 0:
      print ('the list contains an even number')
      break
else:
  print ('the list does not contain even number')

# Using enumerate() function to iterate through a list with indices
# This line creates an enumeration object for the list 'names' which contains index-value pairs.
names = ["Rishu", 'Aayush', 'Ram', 'Shubh']
list(enumerate(names))

# Using enumerate() function to iterate through a list and print index and value pairs
# This loop iterates through each element in the list 'names' along with its index.
# It prints the index and value of each element in the format specified.
names = ["Rishu", 'Aayush', 'Shubh', 'Ram']
for i, j in enumerate(names):
  print('index is {0} and value of index is {1}'.format(i, j))



# While Loop
# To apply 'while' loop we use keyword **while** with the condition and end
# line with a colon.

# We also need to specify an iteration based on a condition which will make
# the 'while' loop stop otherwise 'while' loop will continue exeucting. 
# Example of using a 'while' loop to print numbers from 0 to 9
# This loop iterates from 0 to 9, printing each number along the way.
i = 0
while i < 10:
  print('Number is ', i)
  i = i + 1
  

# Example of using a 'while' loop with a pass statement
# This loop iterates from 0 to 9 but does nothing within the loop body.
# The 'pass' statement is used as a placeholder for future code.
i = 0
while i < 10:
  pass
  i = i + 1


# Example of using a 'while' loop with a continue statement
# This loop iterates from 0 to 9, skipping the number 5 when encountered, using
# the 'continue' statement.
i = 0
while i < 10:
  i = i + 1
  if i == 5:
    continue
  print('Number is ',i)


# Example of using a 'while' loop with a break statement
# This loop iterates from 0 to 9, but it breaks when the number 5 is encountered.
# Hence, it only prints numbers from 0 to 4.
i = 0
while i < 10:
  i = i + 1
  if i == 5:
    break
  print('Number is ', i)


# Example of using a 'while' loop with an else statement
# This loop iterates from 0 to 9, printing each number along the way.
# After the loop completes, the 'else' block is executed.
i = 0
while i < 10:
  print('Number is ', i)
  i = i + 1
else:
  print('Loop has ended')


# Nested Loops

# Here we can use loop inside a loop multiple times. 

# Example of nested loops iterating through a list of lists
# This line defines a list of lists 'list1'.
list1 = [[1, 2, 3, 4], [5, 6, 7, 8]]

# Printing the original list 'list1'
print(list1, '\n')

# Nested loop to iterate through each sublist and each element in the sublists
# The outer loop iterates through each sublist in 'list1'.
# The inner loop iterates through each element in the sublist.
# It prints the square of each element along with a message.
# After printing all elements in a sublist, it prints a newline.
for i in list1 :
    for j in i :
        print(f'Square of {j} is', j ** 2)
    print("\n")




# Best Example for nested loop

# Print triangle shaped star

n = 6  # number of star
k = n + 1   # spaces
for i in range(0, n) :
    for j in range(0, k) :
        print(" ", end = "")
    k = k - 1                  # Reducing spaces after each loop
    for j in range (0, i + 1) :      # Taking new loop for printing *
        print("* ", end = "")
    print("\r")  # After completion of printing '*', start again from new line
 