def swap_keys_values(d):
    swapped_dict = {value: key for key, value in d.items()}
    return swapped_dict


input_dict = {'a': 1, 'b': 2, 'c': 3}
swapped_dict = swap_keys_values(input_dict)
print(swapped_dict)  
