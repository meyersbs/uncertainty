from word import *

__all__ = ['Sentence', 'Sentences']


class Sentence(object):
    def __init__(self, lines):
        self.words = Words(lines)
        self.group = 'c'
        for word in self.words.words:
            if word.group == 'u':
                self.group = 'u'
                break


class Sentences(object):
    def __init__(self, lines):
        self.sentences = list()
        for item in lines:
            self.sentences.append(Sentence(item))

    def get_data(self):
        X, y = list(), list()
        for sentence in self.sentences:
            X.append(sentence)
            y.append(sentence.group)
        return X, y
