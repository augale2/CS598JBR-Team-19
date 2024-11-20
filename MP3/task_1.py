import jsonlines
import sys
import torch
import subprocess
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP3/task_1;
#####################################################

def generate_response(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=1024, temperature=0.8, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def download_and_load_model(model_name="deepseek-ai/deepseek-coder-6.7b-instruct"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # We can also choose load_in_8bit for 8-bit quantization
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16  # Use bfloat16 for reduced memory usage
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    # Loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer, model

def parse_java_code(response):
    start_tag = "```java"
    end_tag = "```"

    # Find the last occurrence of [Java Start] and [Java End]
    last_start_idx = response.rfind(start_tag)
    last_end_idx = response.rfind(end_tag)

    # Check if both tags are present and properly ordered
    if last_start_idx == -1 or last_end_idx == -1 or last_end_idx < last_start_idx:
        return None

    # Extract the content between the last [Java Start] and [Java End] tags
    java_code = response[last_start_idx + len(start_tag):last_end_idx].strip()

    # Handle empty or invalid extraction
    if not java_code:
        return None

    return java_code

def evaluate_java_code(java_code, test_code):
    save_file(java_code, "Solution.java")
    save_file(test_code, "Main.java")

    # Compile the Java code
    compile_process = subprocess.run(
        ["javac", "Solution.java", "Main.java"],
        capture_output=True,
        text=True
    )

    # Check for compilation errors
    if compile_process.returncode != 0:
        print("Compilation Error:\n", compile_process.stderr)  # Print compilation error
        return False, "Compilation Error: " + compile_process.stderr

    # Run the compiled Java program
    run_process = subprocess.run(
        ["java", "Main"],
        capture_output=True,
        text=True
    )

    # Check for runtime errors
    if run_process.returncode != 0:
        print("Execution Error:\n", run_process.stderr)  # Print runtime error
        return False, "Execution Failed: " + run_process.stderr

    # Print successful output
    print("Execution Output:\n", run_process.stdout)
    return True, run_process.stdout

def get_vanilla_prompt(code_snippet,prompt_py):
    return f"""\
    You are an AI programming assistant utilizing the DeepSeek Coder model, developed by DeepSeek Company, and your responses are limited to computer science topics. You must refuse to answer questions on politically sensitive, security, or privacy-related topics.

    ### Instruction:
    Please translate the following Python code into Java.
    Make sure the function name follows Java naming conventions by converting the given prompt funtion name from Python's snake_case (e.g., `function_name`) to Java's camelCase (e.g., `functionName`).
    Make sure that the class name is Solution. Do not name the class as Main.
    Make sure that the new Java code is enclosed exactly between ```java and  ``` tags, as shown below:

    ### Python Code:
    {code_snippet}

    ### Prompt:
    {prompt_py}


    ### Expected Output Format:
    ```java
    <Translated Java Code Here>
    ```

    ### Response:
    """

def get_crafted_prompt(code_snippet, prompt_py, declaration):

    declaration_info = (
        f"\n### Declaration Section:\n"
        #f"The following Java declarations are expected after translation:\n{declaration}\n"
        f"The following Java declarations are expected and must be strictly adhered to during translation:\n{declaration}\n"
        if declaration
        else ""
    )

    return f"""\
    You are an AI programming assistant utilizing the DeepSeek Coder model.

    ### Instruction:
    Please translate the following Python code into Java.
    Make sure the function name follows Java naming conventions by converting the given prompt function name from Python's snake_case (e.g., `function_name`) to Java's camelCase (e.g., `functionName`).
    Make sure that the class name is Solution. Do not name the class as Main.
    Make sure that the new Java code is enclosed exactly between ```java and  ``` tags, as shown below.
    Ensure all  loops are properly enclosed with matching opening and closing curly braces.

    ### Python Code:
    {code_snippet}

    ### Prompt:
    {prompt_py}

    {declaration_info}

    ### Expected Output Format:
    ```java
    <Translated Java Code Here>
    ```

    ### Response:
    """

def parse_java_code(response):
    start_tag = "```java"
    end_tag = "```"

    # Find the last occurrence of [Java Start] and [Java End]
    last_start_idx = response.rfind(start_tag)
    last_end_idx = response.rfind(end_tag)

    # Check if both tags are present and properly ordered
    if last_start_idx == -1 or last_end_idx == -1 or last_end_idx < last_start_idx:
        return None

    # Extract the content between the last [Java Start] and [Java End] tags
    java_code = response[last_start_idx + len(start_tag):last_end_idx].strip()

    # Handle empty or invalid extraction
    if not java_code:
        return None

    return java_code

def save_file(content, file_path):
    with open(file_path, 'w', encoding="utf-8") as file:
        file.write(content)

def analyze_results(results):
    """
    Analyzes the results to calculate counts and lists of correct and incorrect task IDs.

    Args:
        results (list): A list of dictionaries, each representing the result of a task.
                        Example:
                        {
                            "task_id": str,
                            "prompt": str,
                            "response": str,
                            "java_code": str,
                            "is_correct": bool,
                            "status": str
                        }

    Returns:
        dict: A summary of the analysis, including counts and lists of task IDs.
    """
    correct_count = 0
    incorrect_count = 0

    correct_task_ids = []
    incorrect_task_ids = []

    for result in results:
        is_correct = result.get("is_correct", False)  # Default to False if not present
        task_id = result.get("task_id", "unknown_task_id")  # Fallback for safety

        if is_correct:
            correct_count += 1
            correct_task_ids.append(task_id)
        else:
            incorrect_count += 1
            incorrect_task_ids.append(task_id)

    return {
        "correct_count": correct_count,
        "correct_task_ids": correct_task_ids,
        "incorrect_count": incorrect_count,
        "incorrect_task_ids": incorrect_task_ids,
    }

def prompt_model(python_path, java_path, output_path, prompt_function, tokenizer, model,vanilla=True):
    results = []

    # Load data from both jsonl files into dictionaries for quick access
    java_tests = {}
    declarations={}

    with jsonlines.open(java_path) as java_reader:
        for obj in java_reader:
            task_id = obj["task_id"]
            # Extract imports from the declaration section
            declaration = obj.get("declaration", "")
            imports = "\n".join([line for line in declaration.splitlines() if line.startswith("import ")])

            # Extract the test code and prepend imports
            test_code = obj["test"]
            full_test_code = f"{imports}\n\n{test_code}"

            # Store the modified test code in the dictionary
            java_tests[task_id] = full_test_code
            declarations[task_id]=declaration

    with jsonlines.open(python_path) as reader:
        for obj in reader:
            task_id = obj["task_id"]
            prompt_code = obj["canonical_solution"]
            prompt_py = obj["prompt"]
            if task_id not in java_tests:
                continue  # Skip if there's no corresponding test

            test_code = java_tests[task_id]

            # Generate the prompt using the provided function
            if(vanilla):
                prompt = get_vanilla_prompt(prompt_code,prompt_py)
            else:
                prompt = get_crafted_prompt(prompt_code,prompt_py,declarations[task_id])




            # Get the model's response
            response = generate_response(tokenizer, model, prompt)


            java_code = parse_java_code(response)

            print("--------------JAVA CODE------------------")
            print(java_code)
            print("-----------------------------------------")

            print("--##########---TEST CODE------########----")
            print(test_code)
            print("-----------------------------------------")

            if java_code is None:
                results.append({
                    "task_id": task_id,
                    "prompt": prompt,
                    "response": response,
                    "java_code": None,
                    "is_correct": False,
                    "status": "Parsing Error"
                })
                continue

            # Evaluate the translated Java code
            is_correct, status = evaluate_java_code(java_code, test_code)

            # Store results
            result = {
                "task_id": task_id,
                "prompt": prompt,
                "response": response,
                "java_code": java_code,
                "is_correct": is_correct,
                "status": status
            }
            results.append(result)

    print(analyze_results(results))
    # Write results to output JSONL file
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(results)


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
    This Python script is to run prompt LLMs for code translation.
    Usage:
    `python3 task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

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

    print("Model: ", model)
    output_file = args[2]
    if_vanilla = args[3] # True or False


    tokenizer, model_1 = download_and_load_model(model_name=model)
    
    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")
    
    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")
    
    vanilla = True if if_vanilla == "True" else False
    java_path='selected_humanevalx_java_69841794489092767809710484565965079173.jsonl'
    dataset = read_jsonl(input_dataset)
    results = prompt_model(python_path=input_dataset,java_path=java_path, output_path=output_file, prompt_function=get_vanilla_prompt if vanilla else get_crafted_prompt, tokenizer=tokenizer, model=model_1, vanilla=vanilla)
    write_jsonl(results, output_file)
