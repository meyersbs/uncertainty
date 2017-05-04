from word import *

__all__ = ['Sentence', 'Sentences']


class Sentence(object):
    def __init__(self, lines):
        self.sent = []
        self.words = Words(lines)

        for word in self.words.words:
            self.sent.append(word.word)

    def get_words(self):
        return self.words.get_words()

    def get_sent(self):
        return " ".join(self.sent)

    def get_features(self):
        return self.words.get_features()

    def get_label(self, binary=True):
        labels = self.words.get_labels(binary=binary)
        if binary:
            for label in labels:
                if label == "X":
                    return "X"
                elif label != "C":
                    return "U"
            return "C"
        else:
            labs = {"C": 0, "U": 0, "I": 0, "N": 0, "E": 0, "D": 0}
            for label in labels:
                if label == "X":
                    return "X"
                else:
                    labs[label] += 1

            if (labs["U"] != 0 or labs["E"] != 0 or labs["I"] != 0 or
                labs["D"] != 0 or labs["N"] != 0):

                labs.pop("C", None)
                max_val = max(labs.values())
                max_keys = []
                for k, v in labs.items():
                    if v == max_val:
                        max_keys.append(k)

                if len(max_keys) == 1:
                    return max_keys[0]
                else:
                    return "U"
            else:
                return "C"


class Sentences(object):
    def __init__(self, lines):
        self.sentences = list()
        for item in lines:
            self.sentences.append(Sentence(item))

    def get_data(self, binary=True):
        X, y = list(), list()
        for sentence in self.sentences:
            X.append(sentence)
            y.append(sentence.get_label(binary=binary))
        return X, y
