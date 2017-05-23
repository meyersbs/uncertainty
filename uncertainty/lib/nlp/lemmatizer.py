import os

from datetime import datetime as dt

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from ... import helpers
from . import postagger, VERBS_PATH

VERBS = helpers.get_verbs(VERBS_PATH)
WORDNET_POS = {
    'N': wordnet.NOUN, 'V': wordnet.VERB, 'J': wordnet.ADJ, 'R': wordnet.ADV
}
MAP = {
    "'m": 'am', "'ll": 'will', "n't": 'not', "'ve": 'have', "'re": 'are'
}

lemmatizer = WordNetLemmatizer()


class Lemmatizer(object):
    """ Interface. """
    def __init__(self, tokens):
        """ Constructor. """
        self.tokens = tokens

    def execute(self):
        """ Raises NotImplementedError. """
        raise NotImplementedError("Lemmatizer is an abstract class. In must "
                                  "be implemented by another class. Try using "
                                  "the NLTKLemmatizer.")


def fix(token, lemma, prev=None, next=None):
    """
    Attempts to fix lemmatization errors with hardcoded rules.
    """
    if not token and not lemma and not prev and not next:
        raise ValueError("Recieved invalid input to lemmatizer.fix()")
    elif token.lower() == "ca":
        if next and next[0] and next[0].lower() == "n't":
            return "can"
        else:
            return lemma.lower()
    elif token.lower() == "as":
        return "as"
    elif token.lower() == "left":
        if prev and prev[1] == wordnet.VERB:
            return "leave"
        else:
            return lemma.lower()
    elif token in MAP:
        return MAP[token]
    elif lemma in VERBS:
        return VERBS[lemma]
    elif token in VERBS:
        return VERBS[token]
    else:
        return lemma.lower()


class NLTKLemmatizer(Lemmatizer):
    """ Implements Lemmatizer. """
    def __init__(self, tokens):
        """ Constructor. """
        super().__init__(tokens)

    def execute(self):
        """ Return a list of all tokens within the specified string. """
        lemmas = []

        tokens = [
                (t, WORDNET_POS.get(p[0], wordnet.NOUN))
                for (t, p) in postagger.PosTagger(self.tokens).execute()
            ]

        for (i, (token, pos)) in enumerate(tokens):
            lemma = lemmatizer.lemmatize(token, pos)
            prev = None if i == 0 else tokens[i - 1]
            next = None if i == len(tokens) - 1 else tokens[i + 1]
            lemmas.append(fix(token, lemma, prev, next))

        return lemmas
