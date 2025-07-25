import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import *
import itertools

def chainer(text, domain="restaurant", model="gpt-4-turbo"):
    """Processes a given text through a series of chained aspect-based sentiment analysis steps.

    This function dynamically creates different processing chains by permuting the order of aspect,
    sentiment, and category extraction. For each permutation, it processes the input text sequentially,
    with each step feeding its output to the next.

    Args:
        text (str): The input text to analyze.
        domain (str, optional): The domain of the text (e.g., "restaurant", "laptop"). Defaults to "restaurant".
        model (str, optional): The language model to use for the analysis. Defaults to "gpt-4-turbo".

    Yields:
        tuple: A tuple containing the final output of the chain and the sequence of agents used.
    """
    AspectBasedSentimentPrompts.model = model
    AspectBasedSentimentPrompts.domain = domain

    a = Aspects
    s = Sentiments
    c = Categories
    r = Relations

    permutations_3 = list(itertools.permutations([a, s, c]))
    permutations = permutations_3

    for perm in permutations:
        seq = "_".join([cl().__class__.__name__ for cl in perm])
        print(seq)
        p0 = perm[0](text=text)
        p0.process()
        p1 = perm[1](previous_element=p0.current_element, message_history=p0.message_history)
        p1.process()
        try:
            p2 = perm[2](previous_element=p1.current_element, message_history=p1.message_history)
            p2.process()
        except IndexError:
            out = run_icl(p1.message_history, model=model)
            yield out, seq
            continue
        try:
            p3 = perm[3](previous_element=p2.current_element, message_history=p2.message_history)
            p3.process()
        except IndexError:
            out = run_icl(p2.message_history, model=model)
            yield out, seq
            continue
        out = run_icl(p3.message_history, model=model)
        yield out, seq

if __name__ == '__main__':
    text = "Try the lobster teriyaki and the rose special roll."
    for i in chainer(text):
        print(i)