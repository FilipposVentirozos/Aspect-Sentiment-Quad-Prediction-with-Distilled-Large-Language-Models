import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import *
import ast
import json
from collections import defaultdict
import tqdm
import time

def parse_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            sentence, quadruples_str = line.strip().split('####')
            quadruples = ast.literal_eval(quadruples_str)
            quadruples = [quad[:4] for quad in quadruples]
            data.append({
                'sentence': sentence,
                'quadruples': quadruples
            })
    return data

def category_polarity_extraction(data):
    out = list()
    for i in data:
        category_polarity_tuples = list()
        for q in i["quadruples"]:
            category_polarity_tuples.append((q[1], q[2]))
        out.append({"sentence": i['sentence'], "category_polarity": category_polarity_tuples})
    return out

def save_as_json(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

def read_previous_json(inference_filename):
    prefix = inference_filename
    files = os.listdir(os.path.join(data_path, "processed"))
    json_files = [file for file in files if file.startswith(prefix) and file.endswith('.json')]
    count_instances = 0
    for json_file in json_files:
        file_path = os.path.join(os.path.join(data_path, "processed"), json_file)
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            count_instances += len(json_data[list(json_data.keys())[0]])
    return count_instances, len(json_files)

def cot_chainer(*, input_file, ground_truth_file, dataset_name, domain, model):
    data = parse_file(input_file)
    save_as_json(data, ground_truth_file)
    print(f"Data has been saved to {ground_truth_file}")
    inference_filename = model + "_agents_" + dataset_name
    instances, file_n = read_previous_json(inference_filename)
    inference_filename = inference_filename + "_" + str(file_n + 1) + ".json"
    logging.basicConfig(filename=os.path.join(data_path, "processed", "log",
                                              ".".join(inference_filename.split('.')[:-1]) + ".log"),
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    Categories.set_categories(dataset_name)
    fill_icl_template_for_cot_agents(dataset_name)
    agents = defaultdict(list)
    print(instances)
    for i in tqdm.tqdm(data[instances:]):
        print(i['sentence'])
        logger.info(i['sentence'])
        for quads, agent_id in chainer(i['sentence'], domain=domain, model=model):
            agents[agent_id].append(list(quads))
            logger.info(quads[1])
        with open(os.path.join(data_path, "processed", inference_filename), 'w') as file:
            json.dump(agents, file, indent=4)
    print(agents)
    with open(os.path.join(data_path, "processed", inference_filename), 'w') as file:
        json.dump(agents, file, indent=4)

def icl(*, input_file, ground_truth_file, dataset_name, model, examples):
    data = parse_file(input_file)
    save_as_json(data, ground_truth_file)
    print(f"Data has been saved to {ground_truth_file}")
    inference_filename = model + "_icl_" + str(examples) + "_" + dataset_name
    instances, file_n = read_previous_json(inference_filename)
    inference_filename = inference_filename + "_" + str(file_n + 1) + ".json"
    print(inference_filename)
    icl_template_ = fill_icl_template_for_icl(dataset_name, number_of_examples=examples)
    system_instruction = """You are a Natural Language Processing assistant, expert in Aspect-Based Sentiment Analysis. 
    Follow the instructions and do what you have been asked without explanations or reasoning."""
    agents = defaultdict(list)
    for i in tqdm.tqdm(data[instances:]):
        for ij in icl_template_:
            print(ij['content'])
            print("\n")
        out = execute_prompt(chat_history=icl_template_, prompt=i['sentence'],
                             system_instruction=system_instruction,
                             model=model)
        agents["agent_icl_" + str(examples)].append(out)
    with open(os.path.join(data_path, "processed", inference_filename), 'w') as file:
        json.dump(agents, file, indent=4)

def preliminary_prompt_experiments(*, input_file, ground_truth_file, dataset_name, model):
    data = parse_file(input_file)
    data = category_polarity_extraction(data)
    print(dataset_name)
    return preliminary_prompt_experiment(data, model)

def task_switch(task, dataset_name, model_path, **kwargs):
    input_file = os.path.join(data_path, "raw", dataset_name, "test.txt")
    ground_truth_file = os.path.join(data_path, "processed", dataset_name + "_ground_truth.json")
    if task == 'cot_chainer':
        cot_chainer(input_file=input_file, ground_truth_file=ground_truth_file, dataset_name=dataset_name,
                    model=model_path, domain=kwargs['domain'])
    elif task == 'icl':
        icl(input_file=input_file, ground_truth_file=ground_truth_file, dataset_name=dataset_name, model=model_path,
            examples=kwargs['examples'])
    elif task == 'preliminary_prompt_experiments':
        preliminary_prompt_experiments(input_file=input_file, ground_truth_file=ground_truth_file,
                                       dataset_name=dataset_name, model=model_path)

if __name__ == "__main__":
    print("Starting the timer...")
    start_time = time.time()
    for model in ["gpt-4o-2024-08-06"]:
        for dataset_name, domain in [("rest15", "restaurant reviews"), ("rest16", "restaurant reviews"), ("laptop_acos", "laptop_reviews"), ('amazon_ff', 'amazon fine foods reviews'), ("hotels", "hotel reviews"), ("shoes", "shoes reviews")]:
            task = "cot_chainer"
            task_switch(task=task, dataset_name=dataset_name, model_path=model, domain=domain)
            task = "icl"
            examples = [0, 2, 10, 20, 50, 100]
            for example in examples:
                task_switch(task=task, dataset_name=dataset_name, model_path=model, examples=example)
    end_time = time.time()
    print(f"Time taken for the code block: {end_time - start_time} seconds")