
import jsonlines
import sys
import random
import re

def read_jsonl(file_path):
    dataset = []
    with jsonlines.open(file_path) as reader:
        for line in reader: 
            dataset.append(line)
    return dataset


input_dataset = sys.argv[1]
dataset = read_jsonl(input_dataset)

for entry in dataset:
    print("************************************************")
    print(entry['test'] + "\n")
    assertions = re.findall(r'assert candidate\((.*?)\)\s*==\s*(True|False|[^\s]+)', entry['test'])
    print(assertions)

    a = random.choice(assertions) # one random assertion from test
    print(f"Random assertion: {a}\n")

    assert_input = a[0]
    assert_output = a[1]
    print(f"if the input is {expected_input}, the expected output should be {expected_output}")

    print("************************************************\n\n\n\n\n\n\n\n\n")