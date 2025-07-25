# Import specific classes/functions to make them available at the package level

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from elements import Aspects, Sentiments, Relations, Categories, AspectBasedSentimentPrompts
from icl import icl_template, run_icl, fill_icl_template_for_cot_agents, fill_icl_template_for_icl
from chainer import chainer
from models import execute_prompt
# from eval import compute_scores
from preliminary_prompt_experiments import preliminary_prompt_experiment_sentiment, preliminary_prompt_experiment_category
import os
import logging

# Initialization code
print("Initializing __init__")
data_path = os.path.join(os.getcwd(), "data")
config_path = os.path.join(os.getcwd(), "config")

# Define what should be available when using `from mypackage import *`
__all__ = ["Aspects", "Sentiments", "Relations", "Categories", "icl_template", "run_icl", "chainer", "execute_prompt",
           "AspectBasedSentimentPrompts", "data_path", "config_path", "logging", "fill_icl_template_for_cot_agents",
           "fill_icl_template_for_icl", "preliminary_prompt_experiment_sentiment", "preliminary_prompt_experiment_category"]
