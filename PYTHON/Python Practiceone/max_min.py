def find_max_min(numbers):
    if not numbers:
        return None, None 
    
    max_num = numbers[0]
    min_num = numbers[0]
    
    for num in numbers[1:]:
        if num > max_num:
            max_num = num
        if num < min_num:
            min_num = num
    
    return max_num, min_num

input_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
max_num, min_num = find_max_min(input_list)
print("Maximum:", max_num)  
print("Minimum:", min_num)  
