import csv
import re

import helpers

__all__ = ['Word', 'Words']


class Word(object):
    @classmethod
    def from_line(cls, line):
        instance = cls()

        components = next(csv.reader([line], delimiter='\t'))

        instance.word = components[1]
        instance.root = components[2]
        instance.pos = components[3]
        instance.chunk = None
        instance.label = components[4]
        instance.label = re.sub(r"^(B-|I-)", "", instance.label).upper()
        instance.multi_label = components[5]
        instance.binary_label = 'C' if instance.label in ['O', 'C'] else 'U'
        instance.features = dict()
        for feature in components[6:]:
            feature = re.sub(r"\|\|", "|", feature)
            index = feature.rindex(':')
            instance.features[feature[:index]] = float(feature[index + 1:])
        instance.context = None

        return instance

    @classmethod
    def from_group(cls, group):
        instance = cls()

        word, root, pos, chunk = group

        instance.word = word
        instance.root = root
        instance.pos = pos
        instance.chunk = chunk
        instance.label = None
        instance.multi_label = None
        instance.binary_label = None
        instance.features = None
        instance.context = None

        return instance

    def get_features(self):
        return self.features if self.features else self._get_features()

    def get_label(self, binary=True):
        return self.binary_label if binary else self.multi_label

    def _get_features(self):
        features = list()

        features.extend(self.get_typeonefeatures())
        features.extend(self.get_typetwofeatures())
        features.extend(self.get_typethreefeatures())
        features.extend(self.get_typefourfeatures())
        features.extend(self.get_typefivefeatures())
        features.extend(self.get_typesixfeatures())
        features.extend(self.get_typesevenfeatures())

        return dict(zip(features, [1.0] * len(features)))

    def get_typeonefeatures(self):
        return ['prefix_{}_{}'.format(i, self.word[:i]) for i in [3, 4, 5]]

    def get_typetwofeatures(self):
        return ['suffix_{}_{}'.format(i, self.word[-i:]) for i in [3, 4, 5]]

    def get_typethreefeatures(self):
        features = ['root_0_{}'.format(self.root)]
        features += [
                'root_{}_{}'.format(offset, word.root)
                for (offset, word) in self.context.items()
                if word is not None
            ]
        return features

    def get_typefourfeatures(self):
        features = [
                'pattern_0_{}'.format(self.pattern),
                'pattern_prefix_{}'.format(self.pattern[0])
            ]
        # TODO: How to avoid hard coding [-1, 1]?
        features += [
                'pattern_{}_{}'.format(offset, word.pattern)
                for (offset, word) in self.context.items()
                if offset in [-1, 1] and word is not None
            ]
        return features

    def get_typefivefeatures(self):
        features = ['pos_0_{}'.format(self.pos)]
        features += [
                'pos_{}_{}'.format(offset, word.pos)
                for (offset, word) in self.context.items()
                if word is not None
            ]
        return features

    def get_typesixfeatures(self):
        features = ['chunk_0_{}'.format(self.chunk)]
        features += [
                'chunk_{}_{}'.format(offset, word.chunk)
                for (offset, word) in self.context.items()
                if word is not None
            ]
        return features

    def get_typesevenfeatures(self):
        features = [
                'R_0_{}_C_0_{}'.format(self.root, self.chunk),
                'R_0_{}_P_0_{}'.format(self.root, self.pos)
            ]
        # TODO: How to avoid hard coding [-1, 1]?
        features += [
                'R_{}_{}_C_0_{}'.format(offset, word.root, self.chunk)
                for (offset, word) in self.context.items()
                if offset in [-1, 1] and word is not None
            ]
        features += [
                'R_{}_{}_P_0_{}'.format(offset, word.root, self.pos)
                for (offset, word) in self.context.items()
                if offset in [-1, 1] and word is not None
            ]
        return features

    @property
    def pattern(self):
        return helpers.get_wordpattern(self.word)

    def __repr__(self):
        return self.word

    def __str__(self):
        return self.word


class Words(object):
    @classmethod
    def from_lines(cls, lines):
        instance = cls()

        instance.words = list()
        for line in lines:
            if line.strip() == '':
                continue
            instance.words.append(Word.from_line(line))

        return instance

    @classmethod
    def from_groups(cls, groups):
        instance = cls()

        instance.words = list()
        for group in groups:
            instance.words.append(Word.from_group(group))
        instance.establish_context()

        return instance

    def establish_context(self):
        for (index, word) in enumerate(self.words):
            word.context = helpers.get_context(index, self.words)

    def get_words(self):
        return self.words

    def get_data(self, binary=True):
        X, y, z = list(), list(), list()
        for word in self.words:
            X.append(word.features)
            y.append(word.binary_label if binary else word.multi_label)
            z.append(word.word)
        return X, y, z

    def get_features(self):
        features = []
        for word in self.words:
            features.append(word.get_features())
        return features

    def get_labels(self, binary=True):
        labels = []
        for word in self.words:
            labels.append(word.get_label(binary=binary))

        return labels
