import json
import ast
import os
import warnings
from collections import defaultdict
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import *

import warnings
warnings.filterwarnings("ignore")


def read_sentences(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def read_agents_out(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def read_multiple_json(inference_filename):
    prefix = inference_filename
    files = os.listdir(os.path.join(data_path, "processed"))
    json_files = [file for file in files if file.startswith(prefix) and file.endswith('.json')]

    # json_data_list = []
    count_instances = 0
    data = defaultdict(list)
    for file_id, _ in enumerate(range(len(json_files)), start=1):
        file_path = os.path.join(os.path.join(data_path, "processed"),
                                 inference_filename + "_" + str(file_id) + ".json")
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            # print(len(json_data['Aspects_Sentiments_Categories']))
            for k, v in json_data.items():
                data[k].extend(v)
            # json_data_list.append(json_data)
    # keys = ['Aspects_Sentiments_Categories', 'Aspects_Categories_Sentiments', 'Sentiments_Aspects_Categories',
    #         'Sentiments_Categories_Aspects', 'Categories_Aspects_Sentiments', 'Categories_Sentiments_Aspects']
    # fd = dict()
    # for k in keys:
    #     fd[k] = data[k]
    return data


def calculate_tp_fp_fn_asqp(gold_quadruples, pred_quadruples):
    gold_set = set(tuple(q) for q in gold_quadruples)
    try:
        pred_set = set(tuple(q) for q in pred_quadruples)
    except TypeError:
        warnings.warn("Parsing Error")
        pred_set = set()
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def calculate_tp_fp_fn_aste(gold_quadruples, pred_quadruples):
    gold_set = set(tuple(q) for q in gold_quadruples)
    try:
        pred_set = set(tuple(q) for q in pred_quadruples)
    except TypeError:
        warnings.warn("Parsing Error")
        pred_set = set()
    gold_set = {(a, p, o) for a, c, p, o in gold_set}
    if pred_set:
        pred_set = {pred_ for pred_ in pred_set if len(pred_) == 4}
        pred_set = {(a, p, o) for a, c, p, o in pred_set}
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def calculate_tp_fp_fn_acsa(gold_quadruples, pred_quadruples):
    gold_set = set(tuple(q) for q in gold_quadruples)
    try:
        pred_set = set(tuple(q) for q in pred_quadruples)
    except TypeError:
        warnings.warn("Parsing Error")
        pred_set = set()
    gold_set = {(c, p) for a, c, p, o in gold_set}
    if pred_set:
        pred_set = {pred_ for pred_ in pred_set if len(pred_) == 4}
        pred_set = {(c, p) for a, c, p, o in pred_set}
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def string_parsing(quadruples_):
    quad_list = list()
    if quadruples_ == "[]":
        return quad_list
    quadruples_ = quadruples_.strip()
    quadruples_ = quadruples_.replace('\n', " ")
    # quadruples_ = quadruples_.replace('  ', " ")
    quadruples_ = ' '.join(quadruples_.split())
    quads = quadruples_.split("], [")
    quads = [quad.replace("[[", "") for quad in quads]
    quads = [quad.replace("]]", "") for quad in quads]
    for quad in quads:
        elements = list()
        quad = quad.replace('"', "'")

        elements = quad.split("', '")
        # elements = [element.split("', \"") for element in elements]
        # elements = [element.split("\", '") for element in elements]
        # elements = [element.split("\", \"") for element in elements]
        # elements = [element.strip('"') for element in elements]
        elements = [element.strip("'") for element in elements]
        elements = [element.strip('\\') for element in elements]
        quad_list.append(elements)
        # if len(elements) != 4:
        #     print("fefe")
    return quad_list


def calculate_fscore(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fscore = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, fscore


def main(gold_file, pred_file=None, pred_prefix=None):
    gold_sentences = read_sentences(gold_file)
    if pred_file:
        data = read_agents_out(pred_file)
    elif pred_prefix:
        data = read_multiple_json(pred_prefix)
    else:
        raise ValueError("Either pred_prefix or pred_file must be specified")
    # Extract only the amount of labelled data
    gold_sentences = gold_sentences[:len(data[list(data.keys())[0]])]
    # print("Amount of sentences: ", len(data[list(data.keys())[0]]))

    scoring = dict()
    agent_aggregate = defaultdict(list)
    for k, v in data.items():
        gibrish_count = 0
        total_tp_asqp = total_fp_asqp = total_fn_asqp = 0
        total_tp_aste = total_fp_aste = total_fn_aste = 0
        total_tp_acsa = total_fp_acsa = total_fn_acsa = 0
        counts = 0
        for gold_sentence, pred_sentence in zip(gold_sentences, v):
            try:
                try:
                    if "agents" in (pred_prefix or pred_file):
                        quads_pred = string_parsing(pred_sentence[0])  # ast.literal_eval(pred_sentence[0])
                    else:
                        quads_pred = string_parsing(pred_sentence)  # ast.literal_eval(pred_sentence)
                except TypeError:
                    quads_pred = string_parsing(pred_sentence)  # ast.literal_eval(pred_sentence)
            except SyntaxError:
                # print(pred_sentence)
                quads_pred = list()
                gibrish_count += 1
                if "agents" in (pred_prefix or pred_file):
                    warnings.warn(pred_sentence[0])
                else:
                    warnings.warn(pred_sentence)
            for q in quads_pred:
                if len(q) != 4:
                    gibrish_count += 1
                    if "agents" in (pred_prefix or pred_file):
                        warnings.warn(pred_sentence[0])
                    else:
                        warnings.warn(pred_sentence)
            if "agents" in (pred_prefix or pred_file):
                agent_aggregate[k].append(quads_pred)
            # print(quads_pred)
            # print(gold_sentence['quadruples'])
            # print()
            tp, fp, fn = calculate_tp_fp_fn_asqp(gold_sentence['quadruples'], quads_pred)
            total_tp_asqp += tp
            total_fp_asqp += fp
            total_fn_asqp += fn

            tp, fp, fn = calculate_tp_fp_fn_aste(gold_sentence['quadruples'], quads_pred)
            total_tp_aste += tp
            total_fp_aste += fp
            total_fn_aste += fn

            tp, fp, fn = calculate_tp_fp_fn_acsa(gold_sentence['quadruples'], quads_pred)
            total_tp_acsa += tp
            total_fp_acsa += fp
            total_fn_acsa += fn

            # counts += 1
            # if counts == 41:
            #     break
            # print(f"Sentence: {gold_sentence['sentence']}")
            # sentence_precision, sentence_recall, sentence_fscore = calculate_fscore(tp, fp, fn)
            # print(f"  Precision: {sentence_precision:.4f}")
            # print(f"  Recall: {sentence_recall:.4f}")
            # print(f"  F-score: {sentence_fscore:.4f}")
            # print()

        overall_precision, overall_recall, overall_fscore = calculate_fscore(total_tp_asqp, total_fp_asqp,
                                                                             total_fn_asqp)
        scoring[k] = {"ASQP_precision": overall_precision, "ASQP_recall": overall_recall, "ASQP_fscore": overall_fscore}

        overall_precision, overall_recall, overall_fscore = calculate_fscore(total_tp_aste, total_fp_aste,
                                                                             total_fn_aste)
        scoring[k].update(
            {"ASTE_precision": overall_precision, "ASTE_recall": overall_recall, "ASTE_fscore": overall_fscore})

        overall_precision, overall_recall, overall_fscore = calculate_fscore(total_tp_acsa, total_fp_acsa,
                                                                             total_fn_acsa)
        scoring[k].update(
            {"ACSA_precision": overall_precision, "ACSA_recall": overall_recall, "ACSA_fscore": overall_fscore})

    sorted_data = dict(sorted(scoring.items(), key=lambda item: item[1]['ASQP_fscore']))
    for k, v in sorted_data.items():
        print(k)
        print("Overall Scores for ASQP:")
        # print(f"  Precision: {v['ASQP_precision']:.4f}")
        # print(f"  Recall: {v['ASQP_recall']:.4f}")
        print(f"  F-score: {v['ASQP_fscore']:.4f}")
        # print("Overall Scores for ASTE:")
        # print(f"  Precision: {v['ASTE_precision']:.4f}")
        # print(f"  Recall: {v['ASTE_recall']:.4f}")
        # print(f"  F-score: {v['ASTE_fscore']:.4f}")
        # print("Overall Scores for ACSA:")
        # print(f"  Precision: {v['ACSA_precision']:.4f}")
        # print(f"  Recall: {v['ACSA_recall']:.4f}")
        # print(f"  F-score: {v['ACSA_fscore']:.4f}")

        # print("Amount of gibrish generations: ", gibrish_count)
        print("\n")
    # if "agents" in (pred_prefix or pred_file):
    #     print("Agent Aggregate")
    #     aggregate = list()
    #     for i in range(len(agent_aggregate[list(agent_aggregate.keys())[0]])):
    #         agent_res = list()
    #         for agent in agent_aggregate.keys():
    #             agent_res.extend(agent_aggregate[agent][i])
    #         aggregate.append(agent_res)
    #     total_tp_asqp = total_fp_asqp = total_fn_asqp = 0
    #     total_tp_aste = total_fp_aste = total_fn_aste = 0
    #     total_tp_acsa = total_fp_acsa = total_fn_acsa = 0
    #     for gold_sentence, pred_sentence in zip(gold_sentences, aggregate):
    #         tp, fp, fn = calculate_tp_fp_fn_asqp(gold_sentence['quadruples'], pred_sentence)
    #         total_tp_asqp += tp
    #         total_fp_asqp += fp
    #         total_fn_asqp += fn
    #         tp, fp, fn = calculate_tp_fp_fn_aste(gold_sentence['quadruples'], pred_sentence)
    #         total_tp_aste += tp
    #         total_fp_aste += fp
    #         total_fn_aste += fn
    #         tp, fp, fn = calculate_tp_fp_fn_acsa(gold_sentence['quadruples'], pred_sentence)
    #         total_tp_acsa += tp
    #         total_fp_acsa += fp
    #         total_fn_acsa += fn

    #     overall_precision, overall_recall, overall_fscore = calculate_fscore(total_tp_asqp, total_fp_asqp,
    #                                                                          total_fn_asqp)
    #     print("Overall Scores for ASQP:")
    #     print(f"  Precision: {overall_precision:.4f}")
    #     print(f"  Recall: {overall_recall:.4f}")
    #     print(f"  F-score: {overall_fscore:.4f}")

    #     overall_precision, overall_recall, overall_fscore = calculate_fscore(total_tp_aste, total_fp_aste,
    #                                                                          total_fn_aste)

    #     print("Overall Scores for ASTE:")
    #     print(f"  Precision: {overall_precision:.4f}")
    #     print(f"  Recall: {overall_recall:.4f}")
    #     print(f"  F-score: {overall_fscore:.4f}")

    #     overall_precision, overall_recall, overall_fscore = calculate_fscore(total_tp_acsa, total_fp_acsa,
    #                                                                          total_fn_acsa)
    #     print("Overall Scores for ACSA:")
    #     print(f"  Precision: {overall_precision:.4f}")
    #     print(f"  Recall: {overall_recall:.4f}")
    #     print(f"  F-score: {overall_fscore:.4f}")


# Example usage
if __name__ == "__main__":

    print("_____Model: GPT_____\n")

    datasets  = ["shoes", "rest15", "rest16", 'hotels', 'amazon_ff', "laptop_acos", "shoes"]
    # tasks =  ["icl_0", "icl_2", "icl_10", "icl_20",  "icl_50", "icl_100"] 
    # # tasks = ['agents']

    experiments_todo = list()
    # for dataset in datasets:
    #     print("___ Dataset: ", dataset, "___")
    #     for task in tasks:
    #         try:
    #             model = "gpt-4o"
    #             pred_prefix = model + "_" + task + "_" + dataset
    #             gold_file = os.path.join(data_path, "processed", dataset + "_ground_truth.json")
    #             main(gold_file, pred_prefix=pred_prefix, pred_file=None)
    #         except IndexError:
    #             try:
    #                 model = "gpt-4o-2024-08-06"
    #                 pred_prefix = model + "_" + task + "_" + dataset
    #                 gold_file = os.path.join(data_path, "processed", dataset + "_ground_truth.json")
    #                 main(gold_file, pred_prefix=pred_prefix, pred_file=None)
    #             except:
    #                 experiments_todo.append((task, dataset))
            
    #     print()

    # print(experiments_todo)


    tasks = ["icl_20"]
    for dataset in datasets:
        print("___ Dataset: ", dataset, "___")
        for task in tasks:
            print("Chat-Based")
            try:
                model = "gpt-4o"
                pred_prefix = model + "_" + task + "_" + dataset
                gold_file = os.path.join(data_path, "processed", dataset + "_ground_truth.json")
                main(gold_file, pred_prefix=pred_prefix, pred_file=None)
            except IndexError:
                try:
                    model = "gpt-4o-2024-08-06"
                    pred_prefix = model + "_" + task + "_" + dataset
                    gold_file = os.path.join(data_path, "processed", dataset + "_ground_truth.json")
                    main(gold_file, pred_prefix=pred_prefix, pred_file=None)
                except:
                    experiments_todo.append((task, dataset))
            
            print("Single Prompt")
            model = "gpt-4o-2024-08-06"
            pred_prefix = "ASQP_" + model + "_" + task + "_" + dataset
            gold_file = os.path.join(data_path, "processed", dataset + "_ground_truth.json")
            main(gold_file, pred_prefix=pred_prefix, pred_file=None)

        print()
    print(experiments_todo)