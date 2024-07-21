def write_list_to_file(file_path, lines):
    try:
        with open(file_path, 'w') as file:
            for line in lines:
                file.write(line + '\n')
    except IOError:
        print(f"An error occurred while writing to the file at {file_path}.")
lines_to_write = ['First line', 'Second line', 'Third line']
file_path = 'output.txt'
write_list_to_file(file_path, lines_to_write)
