def check_number(num):
    return num > 0 and num % 2 == 0 and num < 100
print(check_number(50))  
print(check_number(101)) 
print(check_number(-2))  
print(check_number(99))  
print(check_number(2))  
