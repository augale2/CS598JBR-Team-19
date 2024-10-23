import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import pytest
import json
import subprocess
import tempfile

def clean_response(response):
    """Clean the response by removing extra whitespace and duplicate content."""
    # Keep all non-empty lines to avoid losing test functions
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    return '\n'.join(lines)

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def get_vanilla_prompt(entry):
    return f"""You are an AI programming assistant. Generate a pytest test suite for the following code.
Only write unit tests in the output and nothing else. Do not simply include test cases provided in the input.

Here is the code to test:

{entry['prompt']}
{entry['canonical_solution']}"""

def get_crafted_prompt(entry):
    return f"""You are an AI programming assistant. Generate a comprehensive pytest test suite including at least 10 test cases for the following Python function that achieves maximum code coverage. 
Include tests for:
1. Edge cases (empty inputs, boundary values, null/None)
2. Typical use cases (normal inputs and expected outputs)
3. Corner cases (minimum/maximum values, type boundaries)
4. All possible execution paths through the code
5. Full branch coverage for all if/else statements
6. Complete loop coverage for all loops
7. Error cases and exception handling

Function to test:

{entry['prompt']}
{entry['canonical_solution']}

Existing test cases to complement:
{entry['test']}

Important:
- Write only pytest test functions
- Each test should focus on one specific scenario
- Use descriptive test names that explain the test case
- Include assertions that verify the exact expected output
- Do not include any explanatory text, only the test code"""

def extract_test_content(response):
    """Extract the test content from the model's response."""
    # Look for pytest content between common markers
    if "def test_" in response:
        # Find the first test function and keep everything after it
        start_idx = response.find("def test_")
        return response[start_idx:]
    return response

def get_coverage(test_content, func_content, task_id):
    # Create temporary test and function files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the function code
        func_name = task_id.split('/')[1]
        func_file = os.path.join(tmpdir, f"task_{func_name}.py")
        with open(func_file, 'w') as f:
            f.write(func_content)
        
        # Save the test code
        test_file = os.path.join(tmpdir, f"test_{func_name}.py")
        with open(test_file, 'w') as f:
            f.write(f"import sys\nsys.path.append('{tmpdir}')\n")
            f.write(f"from task_{func_name} import *\n")
            f.write(test_content)
        
        # Create coverage directory if it doesn't exist
        os.makedirs("Coverage", exist_ok=True)
        
        coverage_file = os.path.join("Coverage", f"{task_id.replace('/', '_')}_report.json")
        
        try:
            # Run pytest with coverage
            result = subprocess.run([
                "pytest",
                test_file,
                f"--cov={tmpdir}",
                "--cov-report",
                f"json:{coverage_file}"
            ], capture_output=True, text=True)
            
            # Read coverage report
            if os.path.exists(coverage_file):
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    file_coverage = coverage_data.get('files', {}).get(func_file, {})
                    coverage_percent = file_coverage.get('summary', {}).get('percent', 0)
                    return coverage_percent, result.stdout, result.stderr
            return 0, result.stdout, result.stderr
        except Exception as e:
            print(f"Error during coverage calculation: {str(e)}")
            return 0, "", str(e)

def prompt_model(dataset, model_name="deepseek-ai/deepseek-coder-6.7b-instruct", vanilla=True):
    print(f"Working with {model_name} prompt type {vanilla}...")
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    results = []
    for entry in dataset:
        try:
            # Create appropriate prompt based on vanilla/crafted setting
            prompt = get_vanilla_prompt(entry) if vanilla else get_crafted_prompt(entry)
            
            # Generate response from model
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                inputs.input_ids,
                max_length=2048,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
    
            raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean the response
            cleaned_response = clean_response(raw_response)
            
            # Extract test content
            test_content = extract_test_content(cleaned_response)
            
            # Create function content
            func_content = entry['prompt'] + entry['canonical_solution']
            
            # Get coverage metrics
            coverage_percent, stdout, stderr = get_coverage(test_content, func_content, entry['task_id'])
            
            print(f"Task_ID {entry['task_id']}:")
            print(f"prompt:\n{prompt}\n")
            print(f"response:\n{test_content}\n")
            print(f"coverage: {coverage_percent}%\n")
            
            results.append({
                "task_id": entry["task_id"],
                "prompt": prompt,
                "response": test_content,
                "coverage": coverage_percent
            })
        except Exception as e:
            print(f"Error processing task {entry['task_id']}: {str(e)}")
            results.append({
                "task_id": entry["task_id"],
                "prompt": prompt if 'prompt' in locals() else "",
                "response": str(e),
                "coverage": 0
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
            f.write(item)

if __name__ == "__main__":
    args = sys.argv[1:]
    input_dataset = args[0]
    model = args[1]
    output_file = args[2]
    if_vanilla = args[3]  # True or False
    
    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a .jsonl file!")
    
    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a .jsonl file!")
    
    vanilla = True if if_vanilla == "True" else False
    
    dataset = read_jsonl(input_dataset)
    results = prompt_model(dataset, model, vanilla)
    write_jsonl(results, output_file)
