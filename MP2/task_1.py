import jsonlines
import sys
import torch
import random
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
    print(f"Working with {model_name} prompt type {vanilla}...")
    
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
    for entry in dataset:

        assert_input = None
        assert_output = None
        
        # randomly pull one of the assertions
        assertions = re.findall(r'assert\s*', entry['test'])
        a = random.choice(assertions) # one random assertion from test

        separation_pattern = r"assert candidate\((.*?)\) == (.*)"
        match = re.search(separation_pattern, a)
        
        print(match)
        if match:
            input_value = match.group(1)  # Extracted input inside candidate()
            expected_value = match.group(2)  # Extracted expected output
            
            print(f"Input: {input_value}")
            print(f"Expected: {expected_value}")

            # Convert the extracted strings into Python objects using eval
            try:
                assert_input = eval(input_value)
                assert_output = eval(expected_value)
            except Exception as e:
                print(f"Error evaluating the assertion: {e}")

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
                     {entry['canonical_solution']}
                     """
        else:
            prompt = f"""
                     ### Context:
                     You are an AI programming assistant. You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer. 

                     ### Instruction:
                     If the string is '{assert_input}', what will the following code return?

                     ### Context:
                     Here is a description on how this code functions:
                     {entry['prompt']}

                     The return value prediction must be enclosed between [Output] and [/Output] tags. For example : [Output]prediction[/Output]
                     You should reason through each step of the program to ensure accuracy

                     ### Code:
                     {entry['canonical_solution']}
                     """
        
        # TODO: prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=1000, temperature=0)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # TODO: process the response and save it to results
        pattern = r"\[Output\](.*?)\[/Output\]" # pull out the prediction
        match = re.search(pattern, response)
        verdict = False

        if match:
            response_prediction = match.group(1) # pull out output
            # Convert the extracted strings into Python objects using eval
            try:
                eval_response_prediction = eval(response_prediction) 
                verdict = assert_output == eval_response_prediction # check if the prediction is correct
            except Exception as e:
                print(f"Error comparing results: {e}")

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_correct:\n{verdict}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "is_correct": verdict
        })
        
    return results

def read_jsonl(file_path):
    dataset = []
    with jsonlines.open(file_path) as reader:
        for line in reader: 
            dataset.append(line)
    return dataset

def write_jsonl(results, file_path):
    with jsonlines.open(file_path, "w") as f:
        for item in results:
            f.write_all([item])

if __name__ == "__main__":
    """
    This Python script is to run prompt LLMs for code synthesis.
    Usage:
    `python3 Task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

    Inputs:
    - <input_dataset>: A `.jsonl` file, which should be your team's dataset containing 20 HumanEval problems.
    - <model>: Specify the model to use. Options are "deepseek-ai/deepseek-coder-6.7b-base" or "deepseek-ai/deepseek-coder-6.7b-instruct".
    - <output_file>: A `.jsonl` file where the results will be saved.
    - <if_vanilla>: Set to 'True' or 'False' to enable vanilla prompt
    
    Outputs:
    - You can check <output_file> for detailed information.
    """
    args = sys.argv[1:]
    input_dataset = args[0]
    model = args[1]
    output_file = args[2]
    if_vanilla = args[3] # True or False
    
    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")
    
    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")
    
    vanilla = True if if_vanilla == "True" else False
    
    dataset = read_jsonl(input_dataset)
    results = prompt_model(dataset, model, vanilla)
    write_jsonl(results, output_file)