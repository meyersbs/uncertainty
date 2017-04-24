__all__ = ['Word', 'Words']


class Word(object):
    def __init__(self, line):
        components = line.split('\t')
        self.word = components[1]
        self.stem = components[2]
        self.pos = components[3]
        self.label = components[4]
        if self.label.startswith('B-') or self.label.startswith('I-'):
            self.label = self.label.replace('B-', '').replace('I-', '')
        self.group = 'c' if self.label == 'O' else 'u'

        self.features = dict()
        for feature in components[5:]:
            index = feature.rindex(':')
            self.features[feature[:index]] = float(feature[index + 1:])


class Words(object):
    def __init__(self, lines):
        self.words = list()
        for line in lines:
            if line.strip() == '':
                continue
            self.words.append(Word(line))

    def get_data(self, binary=True):
        X, y, z = list(), list(), list()
        for word in self.words:
            X.append(word.features)
            if binary:
                y.append(word.group)
            else:
                y.append(word.label)
            z.append(word.word)
        return X, y, z
