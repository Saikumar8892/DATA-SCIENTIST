def sum_of_elements(numbers):
    total = 0
    for number in numbers:
        total += number
    return total
input_list = [1, 2, 3, 4, 5]
total_sum = sum_of_elements(input_list)
print(f"The sum of all elements is: {total_sum}") 
