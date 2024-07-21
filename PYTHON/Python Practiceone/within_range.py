def is_within_range(number, start, end):
    return start <= number <= end


num = 5
start_range = 1
end_range = 10

result = is_within_range(num, start_range, end_range)
print(f"Is {num} within range {start_range} to {end_range}? {result}")  # Output: True

num = 15
result = is_within_range(num, start_range, end_range)
print(f"Is {num} within range {start_range} to {end_range}? {result}")  # Output: False
