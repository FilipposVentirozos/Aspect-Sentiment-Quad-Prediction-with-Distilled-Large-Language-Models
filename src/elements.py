import os
import toml
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import *
from models import execute_prompt

class AspectBasedSentimentPrompts:
    """Base class for handling aspect-based sentiment analysis prompts."""
    model = "gpt-4-turbo"
    domain = "restaurant"

    def __init__(self, text=None, previous_element=None, message_history=None):
        """Initializes the prompt handler.

        Args:
            text (str, optional): The input text to analyze. Defaults to None.
            previous_element (str, optional): The previous element in the processing chain. Defaults to None.
            message_history (list, optional): The history of messages in the conversation. Defaults to None.
        """
        self.previous_element = previous_element
        if message_history:
            self.message_history = message_history
        else:
            self.history_init()
        self.text = text

    def _execute_prompt(self, prompt):
        """Executes a prompt and returns the response.

        Args:
            prompt (str): The prompt to execute.

        Returns:
            str: The response from the language model.
        """
        self.update_history_user(prompt)
        out = execute_prompt(chat_history=self.message_history[1:-1], prompt=self.message_history[-1]['content'],
                                  system_instruction=self.message_history[0]['content'],
                                  model=AspectBasedSentimentPrompts.model)
        self.update_history_assistant(out)
        return out

    def update_history_user(self, message):
        """Adds a user message to the conversation history."""
        self.message_history.append({"role": "user", "content": message})

    def update_history_assistant(self, message):
        """Adds an assistant message to the conversation history."""
        self.message_history.append({"role": "assistant", "content": message})

    def history_init(self):
        """Initializes the conversation history with a system message."""
        template = ("You are a Natural Language Processing assistant, expert in Aspect-Based Sentiment Analysis. "
                    "I want you to force yourself to pick words that you are being asked and only them, "
                    "without explanations or reasoning. If you are unsure, put the most probable.\n")
        self.message_history = [{"role": "user", "content": template}]

class Aspects(AspectBasedSentimentPrompts):
    """Handles the extraction of aspects from a text."""
    def __init__(self, text=None, previous_element=None, message_history=None):
        """Initializes the aspect extractor."""
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
        """Generates the prompt for aspect extraction."""
        return self.prompt_template.format()

    def process(self):
        """Processes the text to extract aspects."""
        prompt = self.generate_prompt()
        return self._execute_prompt(prompt)

class Sentiments(AspectBasedSentimentPrompts):
    """Handles the extraction of sentiments from a text."""
    def __init__(self, text=None, previous_element=None, message_history=None):
        """Initializes the sentiment extractor."""
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
        """Generates the prompt for sentiment extraction."""
        return self.prompt_template.format()

    def process(self):
        """Processes the text to extract sentiments."""
        prompt = self.generate_prompt()
        return self._execute_prompt(prompt)

class Relations(AspectBasedSentimentPrompts):
    """Handles the extraction of relations from a text."""
    def __init__(self, text=None, previous_element=None, message_history=None):
        """Initializes the relation extractor."""
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
        """Generates the prompt for relation extraction."""
        return self.prompt_template.format()

    def process(self):
        """Processes the text to extract relations."""
        prompt = self.generate_prompt()
        return self._execute_prompt(prompt)

class Categories(AspectBasedSentimentPrompts):
    """Handles the extraction of categories from a text."""
    categories = None

    @staticmethod
    def set_categories(dataset_name):
        """Sets the categories for a given dataset."""
        config_path = os.path.join(os.getcwd(), "config")
        categories_file = os.path.join(config_path, dataset_name + '_categories.toml')
        with open(categories_file, 'r') as f:
            Categories.categories = toml.load(f).get('categories')

    def __init__(self, text=None, previous_element=None, message_history=None):
        """Initializes the category extractor."""
        super().__init__(text=text, previous_element=previous_element, message_history=message_history)
        self.current_element = "Categories"
        if self.previous_element:
            template = (f"List the categories from the {self.previous_element} detected. "
                        f"The list of possible categories is: {Categories.categories}. Categories:")
        else:
            template = (f"Given the following categories: {Categories.categories} in the {self.domain} domain. "
                        f"Identify which appear in the following in text:\n\n{self.text}\n\nCategories:")
        self.prompt_template = template

    def generate_prompt(self):
        """Generates the prompt for category extraction."""
        return self.prompt_template.format()

    def process(self):
        """Processes the text to extract categories."""
        prompt = self.generate_prompt()
        return self._execute_prompt(prompt)