def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b != 0:
        return a / b
    else:
        return "Error: Division by zero"

def main():
    try:
        num1 = float(input("Enter the first number: "))
        num2 = float(input("Enter the second number: "))
        
        print(f"{num1} + {num2} = {add(num1, num2)}")
        print(f"{num1} - {num2} = {subtract(num1, num2)}")
        print(f"{num1} * {num2} = {multiply(num1, num2)}")
        print(f"{num1} / {num2} = {divide(num1, num2)}")
        
    except ValueError:
        print("Invalid input. Please enter numeric values.")

if __name__ == "__main__":
    main()
