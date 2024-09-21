###################################################################
# This is a list of all commands you need to run for MP2 on Colab.
###################################################################

# TODO: Clone your GitHub repository
! git clone [Your GitHub Link]
%cd [Your GitHub Repo]

# TODO: Replace the file path of selected_humaneval_[seed].jsonl generated in MP1
input_dataset = ""# selected_humaneval_[seed].jsonl

# Set up requirements for model prompting
! bash -x MP2/setup_models.sh

# TODO: add your seed generated in MP1
seed = "<your_seed>"
task_1_json = "task_1_" + seed + ".jsonl"
task_2_json = "task_2_" + seed + ".jsonl"
task_3_json = "task_3_" + seed + ".jsonl"

# Prompt the models, you can create your `MP2/task_1.py, MP2/task_2.py, MP2/task_3.py` by modifying `MP2/task_[ID].py`
! python3 MP2/task_1.py {input_dataset} "deepseek-ai/deepseek-coder-6.7b-instruct" {task_1_json} "True" |& tee task_1_prompt.log
! python3 MP2/task_2.py {input_dataset} "deepseek-ai/deepseek-coder-6.7b-instruct" {task_2_json} "True" |& tee task_2_prompt.log
! python3 MP2/task_3.py {input_dataset} "deepseek-ai/deepseek-coder-6.7b-instruct" {task_3_json} "True" |& tee task_3_prompt.log
! evaluate_functional_correctness {task_3_json} |& tee task_3_evaluate.log

%cd ..

# git push all nessacery files (e.g., *jsonl, *log) to your GitHub repository
