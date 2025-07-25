# from langchain.prompts import PromptTemplate
import os
from openai import OpenAI
import toml
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import *
from models import execute_prompt


class AspectBasedSentimentPrompts:
    model = "gpt-4-turbo"
    domain = "restaurant"

    def __init__(self, text=None, previous_element=None, message_history=None):
        self.previous_element = previous_element
        # self.model = model
        # Pass the initial template if not just append the history
        if message_history:
            self.message_history = message_history
        else:
            self.history_init()
        # self.domain = domain
        self.text = text

    def _execute_prompt(self, prompt):
        # message_in = {"role": "user", "content": prompt}
        # self.message_history.append(message_in)
        self.update_history_user(prompt)
        out = execute_prompt(chat_history=self.message_history[1:-1], prompt=self.message_history[-1]['content'],
                                  system_instruction=self.message_history[0]['content'],
                                  model=AspectBasedSentimentPrompts.model)

        # response = client.chat.completions.create(
        #     model=self.model,
        #     logprobs=True,
        #     top_logprobs=3,
        #     messages=[*self.message_history]
        # )
        # res_dump = response.model_dump()
        # self.update_history_assistant(res_dump['choices'][0]['message']['content'])
        self.update_history_assistant(out)
        return out

    def update_history_user(self, message):
        self.message_history.append({"role": "user", "content": message})

    def update_history_assistant(self, message):
        self.message_history.append({"role": "assistant", "content": message})

    def history_init(self):
        # template = ("You are a Natural Language Processing assistant, expert in Aspect-Based Sentiment Analysis. "
        #             "In the below I want you to force yourself to pick words that you are being asked and only them, "
        #             "without explanations or reasoning. If you cannot find just put the most possible.\n")
        template = ("You are a Natural Language Processing assistant, expert in Aspect-Based Sentiment Analysis. "
                    "I want you to force yourself to pick words that you are being asked and only them, "
                    "without explanations or reasoning. If you are unsure, put the most probable.\n")
        self.message_history = [{"role": "user", "content": template}]
    # def extract_aspects(self, response):
    #     # Implement logic to extract aspects from the response
    #     pass
    #
    # def extract_sentiments(self, response):
    #     # Implement logic to extract sentiments from the response
    #     pass
    #
    # def extract_relations(self, response):
    #     # Implement logic to extract relations from the response
    #     pass
    #
    # def extract_categories(self, response):
    #     # Implement logic to extract categories from the response
    #     pass


class Aspects(AspectBasedSentimentPrompts):
    def __init__(self, text=None, previous_element=None, message_history=None):
        super().__init__(text=text, previous_element=previous_element, message_history=message_history)
        self.current_element = "Aspects"
        if self.previous_element:
            template = (f"List all word sequences that denote or link to an aspect term from the {self.previous_element} "
                        f"detected. Aspects:")
        else:
            template = (f"Given the following text, list all word sequences that denote an aspect term of the {self.domain} "
                        f"domain:\n\n{self.text}\n\nAspects:")
        self.prompt_template = template

    def generate_prompt(self):
        return self.prompt_template.format()

    def process(self):
        prompt = self.generate_prompt()
        return self._execute_prompt(prompt)


class Sentiments(AspectBasedSentimentPrompts):
    def __init__(self, text=None, previous_element=None, message_history=None):
        super().__init__(text=text, previous_element=previous_element, message_history=message_history)
        self.current_element = "Sentiments"
        if self.previous_element:
            template = (f"List all word sequences that denote or link to a sentiment from the {self.previous_element} "
                        f"detected. Sentiments:")
        else:
            template = (
                f"Given the following text, list all word sequences that denote a sentiment of the {self.domain} "
                f"domain:\n\n{self.text}\n\nSentiments:")
        self.prompt_template = template

    def generate_prompt(self):
        return self.prompt_template.format()

    def process(self):
        prompt = self.generate_prompt()
        return self._execute_prompt(prompt)


class Relations(AspectBasedSentimentPrompts):
    def __init__(self, text=None, previous_element=None, message_history=None):
        super().__init__(text=text, previous_element=previous_element, message_history=message_history)
        self.current_element = "Relations"
        if self.previous_element:
            template = (
                f"List all word sequences that denote a relationship expression from the {self.previous_element}"
                f" detected. Relationship Expressions:")
        else:
            template = (f"Given the following text, list all word sequences that are a relationship expression between "
                        f"aspects and sentiments of the {self.domain} domain:\n\n{self.text}\n\nRelationship Expressions:")
        self.prompt_template = template

    def generate_prompt(self):
        return self.prompt_template.format()

    def process(self):
        prompt = self.generate_prompt()
        return self._execute_prompt(prompt)


class Categories(AspectBasedSentimentPrompts):
    categories = None

    @staticmethod
    def set_categories(dataset_name):

        # Categories.categories = CATEGORIES[dataset_name]
        config_path = os.path.join(os.getcwd(), "config")
        categories_file = os.path.join(config_path, dataset_name + '_categories.toml')
        with open(categories_file, 'r') as f:
            Categories.categories = toml.load(f).get('categories')

    def __init__(self, text=None, previous_element=None, message_history=None):
        super().__init__(text=text, previous_element=previous_element, message_history=message_history)
        self.current_element = "Categories"
        # self.categories = CATEGORIES
        if self.previous_element:
            template = (f"List the categories from the {self.previous_element} detected. "
                        f"The list of possible categories is: {Categories.categories}. Categories:")
        else:
            template = (f"Given the following categories: {Categories.categories} in the {self.domain} domain. "
                        f"Identify which appear in the following in text:\n\n{self.text}\n\nCategories:")
        self.prompt_template = template

    def generate_prompt(self):
        return self.prompt_template.format()

    def process(self):
        prompt = self.generate_prompt()
        return self._execute_prompt(prompt)
