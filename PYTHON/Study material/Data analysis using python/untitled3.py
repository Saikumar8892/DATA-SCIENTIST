import re
def replace_special_characters(s):
    pattern = re.compile("[ ,.]")
    result = pattern.sub(":", s)
    return result
test_strings = [
    "Hello, world. This is a test.",
    "Spaces should be replaced.",
]
for s in test_strings:
    print(f"Original: {s}")
    print(f"Modified: {replace_special_characters(s)}\n")
