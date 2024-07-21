import re
def is_valid_string(s):
    pattern = re.compile("^[a-zA-Z0-9]+$")
    if pattern.match(s):
        return True
    else:
        return False
test_strings = ["Hello123", "Hello 123", "Hello_123", "Hello!", "HelloWorld"]
for s in test_strings:
    if is_valid_string(s):
        print(f"'{s}' contains only allowed characters.")
    else:
        print(f"'{s}' contains characters outside the allowed set.")
