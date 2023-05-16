import random

# Read the contents of the transcript file
with open('source.txt', 'r') as file:
    lines = file.readlines()

# Remove quotation marks and newlines, and shuffle the lines
lines = [line.strip('"\n') for line in lines]
# Remove empty lines
lines = [line for line in lines if line.strip() != '']
random.shuffle(lines)

# Write the modified lines to a new shuffled transcript file
with open('source.txt', 'w') as file:
    file.write('\n'.join(lines))
