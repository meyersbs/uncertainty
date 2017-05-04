import re

__all__ = ['Word', 'Words']


class Word(object):
    def __init__(self, line):
        components = line.split('\t')
        self.word = components[1]
        self.stem = components[2]
        self.pos = components[3]
        self.label = components[4]
        self.multi_label = components[5]
        self.label = re.sub(r"^(B-|I-)", "", self.label).upper()
        self.binary_label = 'C' if self.label in ['O', 'C'] else 'U'
        #self.multi_label = self.label if self.label != 'O' else 'C'

        self.features = dict()
        for feature in components[6:]:
            feature = re.sub(r"\|\|", "|", feature)
            index = feature.rindex(':')
            self.features[feature[:index]] = float(feature[index + 1:])

    def get_features(self):
        return self.features

    def get_label(self, binary=True):
        if binary:
            return self.binary_label
        else:
            return self.multi_label

class Words(object):
    def __init__(self, lines):
        self.words = list()
        for line in lines:
            if line.strip() == '':
                continue
            self.words.append(Word(line))

    def get_words(self):
        return self.words

    def get_data(self, binary=True):
        X, y, z = list(), list(), list()
        for word in self.words:
            X.append(word.features)
            if binary:
                y.append(word.binary_label)
            else:
                y.append(word.multi_label)
            z.append(word.word)
        return X, y, z

    def get_features(self):
        feats = []
        for word in self.words:
            feats.append([word, word.get_features()])

        return feats

    def get_labels(self, binary=True):
        labels = []
        for word in self.words:
            labels.append(word.get_label(binary=binary))

        return labels
