def max_of_three(a, b, c):
    if a >= b and a >= c:
        return a
    elif b >= a and b >= c:
        return b
    else:
        return c
num1 = int(input("enter number1:- "))
num2 = int(input("enter number2:- "))
num3 = int(input("enter number3:- "))
max_num = max_of_three(num1, num2, num3)
print(f"The maximum of {num1}, {num2}, and {num3} is {max_num}.")


def sum_of_list(numbers):
    total = 0
    for number in numbers:
        total += number
    return total
numbers_list = [1, 2, 3, 4, 5]
total_sum = sum_of_list(numbers_list)
print(f"The sum of the numbers in the list {numbers_list} is {total_sum}.")


def multiply_list(numbers):
    result = 1
    for number in numbers:
        result *= number
    return result
numbers_list = [1,2,3,4,5]
product = multiply_list(numbers_list)
print(f"the multiply of the numbers in the list{numbers_list} is {product}")
 
