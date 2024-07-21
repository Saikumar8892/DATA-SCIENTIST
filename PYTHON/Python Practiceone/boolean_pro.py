def at_least_two_true(a, b, c):
    return (a and b) or (a and c) or (b and c)
result = at_least_two_true(True, True, False)
print(result) 

result = at_least_two_true(True, False, False)
print(result) 

result = at_least_two_true(True, True, True)
print(result)  

result = at_least_two_true(False, False, False)
print(result) 
