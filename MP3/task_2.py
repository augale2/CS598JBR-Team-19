import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def load_model(model_name):
    """Download and load the model with quantization."""
    print(f"Loading model: {model_name}...")
    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return tokenizer, model

def generate_prompt(entry, vanilla=True):
    """Create a prompt based on whether it's vanilla or crafted."""
    if vanilla:
        prompt = f"""
You are an AI programming assistant. Analyze the following code and determine if it's correct or buggy.

### Instruction:
{entry['instruction']}

### Code:
{entry['buggy_solution']}

Is the above code buggy or correct? Please explain your step by step reasoning. The prediction should be enclosed within a single <start> and <end> tags. For example: <start>Buggy<end>

"""
    else:
        prompt = f"""
You are an AI programming assistant. Analyze the CODE TO ANALYZE solution to determine whether it is buggy or correct. Use the CANONICAL SOLUTION and the TEST CASES for your analysis and provide a single verdict.

PROBLEM:
{entry['instruction']}

CANONICAL SOLUTION:
{entry['canonical_solution']}

CODE TO ANALYZE:
{entry['buggy_solution']}

TEST CASES:
{entry['test']}

Is the above code buggy or correct? Please explain your step by step reasoning. The prediction should be enclosed within a single <start> and <end> tags. For example: <start>Buggy<end>

"""
    return prompt

def get_model_response(tokenizer, model, prompt):
    """Generate a response using the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=1024, temperature=0.2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# def extract_verdict(response):
#     """Extract the verdict from the response."""
#     if "<start>" in response and "<end>" in response:
#         verdict_text = response.split("<start>")[1].split("<end>")[0].strip().lower()
#         print(f"\n\n\n\n{verdict_text} here is printed here")
#         if "buggy" in verdict_text:
#             return "buggy"
#         elif "correct" in verdict_text:
#             return "correct"
#     return None

def extract_verdict(response):
    """Extract the verdict from the response."""
    # Split the response by "<start>" to get all segments
    parts = response.split("<start>")
    
    # Ensure there are at least 3 segments to reach the third occurrence
    if len(parts) >= 4 and "<end>" in parts[3]:
        # Extract the text between the third "<start>" and "<end>"
        verdict_text = parts[3].split("<end>")[0].strip().lower()
        # print(f"\n\n\n\n{verdict_text} is printed here")
        
        if "buggy" in verdict_text:
            return "buggy"
        elif "correct" in verdict_text:
            return "correct"
    return None


def prompt_model(dataset, model_name, vanilla=True):
    """Process dataset, generate prompts, and get responses."""
    tokenizer, model = load_model(model_name)
    
    results = []
    for entry in dataset:
        prompt = generate_prompt(entry, vanilla)
        response = get_model_response(tokenizer, model, prompt)
        model_prediction = extract_verdict(response)
        
        # Determine the ground truth
        ground_truth = "buggy"
        
        # Compare model's prediction with ground truth
        is_correct = model_prediction == ground_truth
        # is_correct = extract_verdict(response) == (entry['bug_type'] == "buggy")
        
        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_expected:\n{is_correct}")
        # print(f"\n{model_prediction} printed")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "is_correct": is_correct
        })
        
    return results

def read_jsonl(file_path):
    """Read the JSONL file and return the dataset."""
    dataset = []
    with jsonlines.open(file_path) as reader:
        for line in reader: 
            dataset.append(line)
    return dataset

def write_jsonl(results, file_path):
    """Write results to a JSONL file."""
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
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")
    
    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")
    
    vanilla = True if if_vanilla == "True" else False
    
    dataset = read_jsonl(input_dataset)
    results = prompt_model(dataset, model, vanilla)
    write_jsonl(results, output_file)
