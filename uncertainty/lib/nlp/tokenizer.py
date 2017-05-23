from nltk.tokenize import word_tokenize


class Tokenizer(object):
    """ Interface. """
    def __init__(self, text):
        """ Constructor. """
        self.text = text

    def execute(self):
        """ Raises NotImplementedError. """
        raise NotImplementedError("Tokenizer is an abstract class. In must be "
                                  "implemented by another class. Try using "
                                  "the NLTKTokenizer.")


class NLTKTokenizer(Tokenizer):
    """ Implements Tokenizer. """
    def __init__(self, text):
        """ Constructor. """
        super(NLTKTokenizer, self).__init__(text)

    def execute(self):
        """ Return a list of all tokens within the specified string. """
        return word_tokenize(self.text)
