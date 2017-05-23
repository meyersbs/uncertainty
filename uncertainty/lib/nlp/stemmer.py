import nltk


class Stemmer(object):
    """
    Given a list of tokens, return a list of stems associated with those tokens
    """
    def __init__(self, tokens):
        """ Constructor. """
        self.stemmer = nltk.PorterStemmer()
        self.tokens = tokens

    def execute(self):
        """
        Given a list of tokens, return a list of stems associated with those
        tokens.
        """
        return [self.stemmer.stem(token).lower() for token in self.tokens]
