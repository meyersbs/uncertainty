"""
@AUTHOR: nuthanmunaiah
"""

import nltk


class PosTagger(object):
    """
    Given a list of tokens, return a list of tuples of the form:
    (token, part-of-speech-tag)
    """
    def __init__(self, tokens):
        """ Constructor. """
        self.tokens = tokens

    def execute(self):
        """
        Given a list of tokens, return a list of tuples of the form:
        (token, part-of-speech-tag)
        """
        return nltk.pos_tag(self.tokens)
