import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import *


def preliminary_prompt_experiment_sentiment(data, model):
    results_sentiment, results_opinion_polarity = list(), list()
    # Generic Prompts
    system_instruction = """Provide answers to only what are being asked succinctly,
     no explanations or reasoning."""
    closing_prompt = "Please provide your answer in only one word between 'positive', 'negative' and 'neutral'"
    for i in tqdm.tqdm(data[:200]):
        if i['category_polarity']:
            for cp in i['category_polarity']:
                category = cp[0]

                # Sentiment
                chat_history = list()
                template_sentiment = f"What is the sentiment of category '{category}' in the text '{i['sentence']}'?"
                out = execute_prompt(chat_history=chat_history, prompt=template_sentiment,
                                     system_instruction=system_instruction, model=model)
                chat_history.append({"role": "user", "content": template_sentiment})
                chat_history.append({"role": "assistant", "content": out})
                out = execute_prompt(chat_history=chat_history, prompt=closing_prompt,
                                     system_instruction=system_instruction, model=model)
                if cp[1].lower() in out.lower().strip():
                    results_sentiment.append(True)
                else:
                    results_sentiment.append(False)

                # Opinion Category
                chat_history = list()
                template_opinion = f"Which is the opinion expressed of category '{category}' in the text '{i['sentence']}'?"
                out = execute_prompt(chat_history=chat_history, prompt=template_opinion,
                                     system_instruction=system_instruction, model=model)
                chat_history.append({"role": "user", "content": template_opinion})
                chat_history.append({"role": "assistant", "content": out})
                template_polarity = f"What is the polarity of the opinion?"
                out = execute_prompt(chat_history=chat_history, prompt=template_polarity,
                                     system_instruction=system_instruction, model=model)
                chat_history.append({"role": "user", "content": template_polarity})
                chat_history.append({"role": "assistant", "content": out})
                out = execute_prompt(chat_history=chat_history, prompt=closing_prompt,
                                     system_instruction=system_instruction, model=model)
                if cp[1].lower() in out.lower():
                    results_opinion_polarity.append(True)
                else:
                    results_opinion_polarity.append(False)

    print("results_sentiment: ", sum(results_sentiment) / len(results_sentiment))
    print("results_opinion_polarity: ", sum(results_opinion_polarity) / len(results_opinion_polarity))

    return results_sentiment, results_opinion_polarity


def calculate_precision_recall(predicted_set, ground_truth_set):
    # True Positives
    true_positives = predicted_set & ground_truth_set  # Intersection
    
    # False Positives
    false_positives = predicted_set - ground_truth_set  # Difference
    
    # False Negatives
    false_negatives = ground_truth_set - predicted_set  # Difference
    
    # Precision calculation
    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0
    
    # Recall calculation
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if (len(true_positives) + len(false_negatives)) > 0 else 0
    
    return precision, recall

def f1_score(precision, recall):
    # Calculate the F1 score
    if precision + recall == 0:
        return 0  # Avoid division by zero
    return 2 * (precision * recall) / (precision + recall)


def preliminary_prompt_experiment_category(data, categories, model):
    categories = [element.lower() for element in categories]

    results_category_precision, results_category_recall, results_term_category_precision, results_term_category_recall = list(), list(), list(), list()
    # Generic Prompts
    system_instruction = """Provide answers to only what are being asked succinctly,
     no explanations or reasoning."""
    # closing_prompt = f"Please provide one answer from the following: {', '.join(map(str, categories))}"
    for i in tqdm.tqdm(data[:200]):
        # if i['category_polarity']:

        current_categories = set()
        for cp in i['category_polarity']:
            current_categories.add(cp[0])

            # category = cp[0]
        # Category
        chat_history = list()
        template_category = f"You are a Natural Language Processing assistant, expert in Aspect-Based Sentiment Analysis. What categories can you detect from the following: {', '.join(map(str, categories))}, in this text '{i['sentence']}'. Please provide the related categories as they are displayed, verbatim, and have them comma separated."
        out = execute_prompt(chat_history=chat_history, prompt=template_category,
                                system_instruction=system_instruction, model=model)
        cats = out.split(",")
        pred_cats = {cat.strip().lower() for cat in cats}
        precision, recall = calculate_precision_recall(pred_cats, current_categories)

        results_category_precision.append(precision)
        results_category_recall.append(recall)


        # Opinion Category
        chat_history = list()
        template_term = f"You are a Natural Language Processing assistant, expert in Aspect-Based Sentiment Analysis. What aspect term can you identify in this text '{i['sentence']}'."
        out = execute_prompt(chat_history=chat_history, prompt=template_term,
                             system_instruction=system_instruction, model=model)
        chat_history.append({"role": "user", "content": template_term})
        chat_history.append({"role": "assistant", "content": out})
        template_category = f"From the extracted aspect terms, what categories can you detect from the following list: {', '.join(map(str, categories))}. Please provide the related categories as they are displayed, verbatim, and have them comma separated."
        out = execute_prompt(chat_history=chat_history, prompt=template_category,
                             system_instruction=system_instruction, model=model)
        cats = out.split(",")
        pred_cats = {cat.strip().lower() for cat in cats}
        precision, recall = calculate_precision_recall(pred_cats, current_categories)

        results_term_category_precision.append(precision)
        results_term_category_recall.append(recall)

    # Calculate the simple average (arithmetic mean)
    avg_precision_category = sum(results_category_precision) / len(results_category_precision)
    avg_recall_category = sum(results_category_recall) / len(results_category_recall)

    print("results_category precision: ", avg_precision_category)
    print("results_category recall: ", avg_recall_category)
    print("results_category f1: ", f1_score(avg_precision_category, avg_recall_category))

    avg_term_category_precision = sum(results_term_category_precision) / len(results_term_category_precision)
    avg_term_category_recall = sum(results_term_category_recall) / len(results_term_category_recall)
    
    print("results_term_category precision: ", avg_term_category_precision)
    print("results_term_category recall: ", avg_term_category_recall)
    print("results_term_category f1: ", f1_score(avg_term_category_precision, avg_term_category_recall))

    # print()
    return avg_precision_category, avg_recall_category, avg_term_category_precision, avg_term_category_recall