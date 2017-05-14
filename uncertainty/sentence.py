from .word import *

__all__ = ['Sentence', 'Sentences']


class Sentence(object):
    @classmethod
    def from_lines(cls, lines):
        instance = cls()

        instance.words = Words.from_lines(lines)

        instance.sentence = list()
        for word in instance.words.words:
            instance.sentence.append(word.word)

        return instance

    @classmethod
    def from_groups(cls, groups):
        instance = cls()

        instance.words = Words.from_groups(groups)

        instance.sentence = list()
        for word in instance.words.words:
            instance.sentence.append(word.word)

        return instance

    def get_words(self):
        return self.words.get_words()

    def get_sentence(self):
        return " ".join(self.sentence)

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

            if (
                    labs["U"] != 0 or labs["E"] != 0 or labs["I"] != 0 or
                    labs["D"] != 0 or labs["N"] != 0
            ):

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

    def get_data(self, binary=True):
        X, y, z = self.words.get_data(binary=True)
        return X, y, z


class Sentences(object):
    @classmethod
    def from_lineslist(cls, lineslist):
        instance = cls()

        instance.sentences = list()
        for lines in lineslist:
            instance.sentences.append(Sentence.from_lines(lines))

        return instance

    def get_data(self, binary=True):
        X, y = list(), list()
        for sentence in self.sentences:
            X.append(sentence)
            y.append(sentence.get_label(binary=binary))
        return X, y
