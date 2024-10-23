import jsonlines
import sys
import torch
import random
import re
import json
import ast
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################
def escape_brackets(input_string):
    # Check if the input string starts with any type of brackets
    if input_string.startswith('[') or input_string.startswith('{'):
        # Escape the brackets
        input_string = input_string.replace('[', r'\[').replace(']', r'\]')
        input_string = input_string.replace('{', r'\{').replace('}', r'\}')
    return input_string


def load_fresh_model(model_name, cache_dir, bnb_config):
    """Download and load a fresh copy of the model"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    return model, tokenizer

def clear_model_cache():
    """Utility function to clear the model cache if needed"""
    import shutil
    cache_dir = "/content/model_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

def parse_json_object_1(data):
    task_id = data.get("task_id", "")
    prompt = data.get("prompt", "").strip()
    canonical_solution = data.get("canonical_solution", "").strip()
    test = data.get("test", "").strip()

    # Create a set to store all valid assertions
    valid_assertions = set()

    # Initialize an empty string to accumulate lines for multi-line assertions
    current_assertion = ""

    # Split the test into lines and process each line
    test_lines = test.split('\n')
    for line in test_lines:
        line = line.strip()

        # Check if the line starts a new assertion
        if line.startswith("assert") and "==" in line:
            # If there's an ongoing assertion, add it to the valid assertions
            if current_assertion:
                valid_assertions.add(current_assertion)
                current_assertion = ""  # Reset for next assertion

            current_assertion = line  # Start a new assertion

        elif current_assertion:  # If we're in a multi-line assertion
            current_assertion += " " + line  # Accumulate the line

    # Add the last accumulated assertion if any
    if current_assertion:
        valid_assertions.add(current_assertion)

    # Initialize the dictionary to store all parsed assertions
    parsed_assertions = {}

    # Process each assertion and store in a dict of dicts
    for idx, assertion in enumerate(valid_assertions, start=1):
        try:
            # Extract the input part using regex to capture everything inside the parentheses
            input_match = re.search(r'assert candidate\((.*?)\)\s*==', assertion)
            if input_match:
                input_part = input_match.group(1).strip()
            else:
                input_part = ""

            # Improved regex for extracting the expected output part
            expected_output_match = re.search(r'==\s*(\[.*?\](?=\s*(?:,|$))|[^,\s]+)', assertion)
            if expected_output_match:
                expected_output = expected_output_match.group(1).strip()
            else:
                expected_output = ""

            # Clean up quotes around input and expected output
            input_part = input_part.strip("'\"")

            # Improved list handling for expected output
            if expected_output.startswith('[') and expected_output.endswith(']'):
                try:
                    # Use ast.literal_eval for safer evaluation of string representations of lists
                    expected_output = ast.literal_eval(expected_output)
                except:
                    # Fallback to manual parsing if literal_eval fails
                    expected_output = expected_output[1:-1].split(',')
                    expected_output = [item.strip().strip("'\"") for item in expected_output]
            elif expected_output == "[]":
                expected_output = []
            else:
                # Handle non-list outputs
                expected_output = expected_output.strip("'\"")

            # Handling input parsing for lists and other types
            if input_part.startswith('[') and input_part.endswith(']'):
                try:
                    # Use ast.literal_eval for safer evaluation of list inputs
                    input_part = ast.literal_eval(input_part)
                except:
                    # Fallback to original parsing method
                    input_part = input_part[1:-1].strip()  # Remove brackets
                    input_items = [item.strip() for item in input_part.split(',')]  # Split items into list
                    # Clean quotes and convert types
                    input_list = []
                    for item in input_items:
                        item = item.strip().strip("'").strip('"')
                        try:
                            if item.isdigit():  # Check if it's a digit
                                input_list.append(int(item))
                            else:
                                input_list.append(item)
                        except ValueError:
                            input_list.append(item)
                    input_part = input_list

            # Store the parsed assertion in the dictionary
            parsed_assertions[f"assertion{idx}"] = {
                "task_id": task_id,
                "prompt": prompt,
                "canonical_solution": canonical_solution,
                "input": input_part,
                "expected_output": expected_output
            }
        except ValueError as e:
            continue

    # Return the dictionary of assertions
    return parsed_assertions if parsed_assertions else None

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)
def construct_few_shot_prompt(parsed_assertions, selected_assertion):
    # Limit the number of examples to 1
    examples = ""
    example_count = 0

    for key, assertion in parsed_assertions.items():
        if example_count >= 4:
            break  # Limit to 4 examples

        example_input = assertion["input"]
        example_expected_output = assertion["expected_output"]

        examples += f"Input: '{example_input}' Expected Output: [Output] {example_expected_output} [/Output] \n"
        example_count += 1

    # Construct the few-shot prompt
    few_shot_prompt = f"""
    ### Context:
    You are an AI programming assistant. You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

    ### Instruction:
    Use the examples below to guide your response.

    ### Examples:
    {examples}

    ### Task:
    Given the input below, provide the correct output:
    Input: '{selected_assertion['input']}'
    Expected Output: [Output] Your Prediction Here [/Output]
    """

    return few_shot_prompt

def analyze_results(results):
    correct_count = 0
    incorrect_count = 0

    correct_task_ids = []
    incorrect_task_ids = []

    for result in results:
        if result.get("is_correct"):
            correct_count += 1
            correct_task_ids.append(result["task_id"])
        else:
            incorrect_count += 1
            incorrect_task_ids.append(result["task_id"])

    return {
        "correct_count": correct_count,
        "correct_task_ids": correct_task_ids,
        "incorrect_count": incorrect_count,
        "incorrect_task_ids": incorrect_task_ids,
    }

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla=True):
    cache_dir = "/content/model_cache"
    os.makedirs(cache_dir, exist_ok=True)
    marker_file = os.path.join(cache_dir, "model_downloaded.txt")
    # TODO: download the model; load the model with quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # We can also choose load_in_8bit for 8-bit quantization
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16  # Use bfloat16 for reduced memory usage
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    # Loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    results = []
    with open(dataset, 'r',encoding='utf-8') as f:
        for entry in f:

            obj = json.loads(entry.strip())  # Parse each JSON object on the line

            #Select a random key and delete that assertion from the dict
            assertion_dict=parse_json_object_1(obj)
            selected_key = random.choice(list(assertion_dict.keys()))
            parsed_result = assertion_dict[selected_key]
            assertion_dict.pop(selected_key)

            assert_input = parsed_result["input"]
            assert_output = parsed_result["expected_output"]

            # TODO: create prompt for the model
            # Tip : Use can use any data from the dataset to create
            #       the prompt including prompt, canonical_solution, test, etc.
            if vanilla:
                prompt = f"""
                        ### Context:
                        You are an AI programming assistant. You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

                        ### Instruction:
                        If the string is '{assert_input}', what will the following code return?
                        The return value prediction must be enclosed between [Output] and [/Output] tags. For example : [Output]prediction[/Output]


                        ### Code:
                        {parsed_result['canonical_solution']}
                            """
            else:
                prompt = construct_few_shot_prompt(assertion_dict,parsed_result)

            # TODO: prompt the model and get the response
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_length=1000, temperature=0)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            response_output=''
            if vanilla:
                match = re.search(r'### Output:\s*\[Output\](.*?)\[/Output\]', response, re.DOTALL)
                if match:
                    response_output=match.group(1).strip()
                else:
                    respone_output=None
            else:
                assert_input1=escape_brackets(str(assert_input))
                pattern = rf"Input:\s*'{assert_input1}'\s*Expected Output:\s*\[Output\]\s*(.*?)\s*\[\/Output\]"

                match = re.search(pattern, response)

                if match:
                    response_output = match.group(1)
                else:
                    response_output = None

            results.append({
                "task_id": parsed_result["task_id"],
                "prompt": prompt,
                "response": response,
                "is_correct": response_output==parsed_result["expected_output"]
                })
    analysis = analyze_results(results)
    print(analysis)
    return results

def write_jsonl(results, file_path):
    with jsonlines.open(file_path, "w") as f:
        for item in results:
            f.write_all([item])



if __name__ == "__main__":
    dataset_path = sys.argv[1]  # Path to dataset
    save_path = sys.argv[3] 

    model = sys.argv[2]

    vanilla = sys.argv[4]    # Path to save the results JSON file
    
    results = prompt_model(dataset_path, model,vanilla)  # Get the model response
    
    print(f"Dataset Path: {dataset_path}")
    print(f"Save Path: {save_path}")
    print(f"Model: {model}")
    print(f"Vanilla Path: {vanilla}")
    
    write_jsonl(results, save_path) # Save the results to a file
