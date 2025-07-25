import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import *

import itertools


def chainer(text, domain="restaurant", model="gpt-4-turbo"):
    AspectBasedSentimentPrompts.model = model  # Set model 4 all
    AspectBasedSentimentPrompts.domain = domain

    a = Aspects
    s = Sentiments
    c = Categories
    r = Relations

    # Generate all possible permutations of the list
    # permutations_4 = list(itertools.permutations([a, s, c, r]))
    permutations_3 = list(itertools.permutations([a, s, c]))
    # permutations_2 = list(itertools.permutations([a, s]))
    permutations = permutations_3  #  permutations_4 + permutations_3 + permutations_2  # 
    # permutations = [[r, s, a, c]]
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
