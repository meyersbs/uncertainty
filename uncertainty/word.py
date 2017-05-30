import csv
import re

from . import helpers
from . import constants

RE_CLASS_PREFIX = re.compile(r'^(B-|I-)')

__all__ = ['Word', 'Words']


class Word(object):
    @classmethod
    def from_line(cls, line):
        instance = cls()

        components = line

        instance.word = components[1]
        instance.root = components[2]
        instance.pos = components[3]
        instance.chunk = None
        instance.label = RE_CLASS_PREFIX.sub('', components[4])
        instance.multi_label = constants.UNCERTAINTY_CLASS_MAP[instance.label]
        instance.binary_label = 'U' if instance.multi_label != 'C' else 'C'
        instance.features = dict()
        for feature in components[5:]:
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
        '''Return prefixes of lengths 3, 4 and 5

        Returns
        ------
        features : list
            A list of prefixes of lengths 3, 4 and 5 of the current word.

        Example
        -------

        If 'Distinct' is the current word, then ['prefix_3_Dis',
        'prefix_4_Dist', 'prefix_5_Disti'] is returned.
        '''
        return ['prefix_{}_{}'.format(i, self.word[:i]) for i in [3, 4, 5]]

    def get_typetwofeatures(self):
        '''Return suffixes of lengths 3, 4 and 5

        Returns
        ------
        features : list
            A list of suffixes of lengths 3, 4 and 5 of the current word.

        Example
        -------

        If 'Distinct' is the current word, then ['suffix_3_nct',
        'suffix_4_inct', 'suffix_5_tinct'] is returned.
        '''
        return ['suffix_{}_{}'.format(i, self.word[-i:]) for i in [3, 4, 5]]

    def get_typethreefeatures(self):
        '''Return root (stem or lemma) of words in a window length of 2

        Returns
        -------
        features : list
            A list containing the root of the current word and that of two
            words before and after the current word.

        Example
        -------

        If 'Regulating' is the current word in the context 'Cells in Regulating
        Cellular Immunity', then ['root_-2_cell', 'root_-1_in',
        'root_0_regulate, 'root_1_cellular', 'root_2_immun'] is returned.
        '''
        features = ['root_0_{}'.format(self.root)]
        features += [
                'root_{}_{}'.format(offset, word.root)
                for (offset, word) in self.context.items()
                if word is not None
            ]
        return features

    def get_typefourfeatures(self):
        '''Return pattern prefixes of words in a window length of 1

        Returns
        -------
        features : list
            A list containing pattern prefix of the current word and that of
            one word before and after the current word. In addition to the
            pattern prefixes, the first character of the pattern prefix of the
            current word is also included in the list.

        Example
        -------

        If 'Regulating' is the current word in the context 'Cells in Regulating
        Cellular Immunity', the ['pattern_-1_a', 'pattern_0_Aa',
        'pattern_1_Aa', 'pattern_prefix_A'] is returned.
        '''
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
        '''Return part of speech tag of words in a window length of 2

        Returns
        -------
        features : list
            A list containing the part of speech tag of the current word and
            that of two words before and after the current word.

        Example
        -------

        If 'Regulating' is the current word in the context 'Cells (NNS) in (IN)
        Regulating (VBG) Cellular (JJ) Immunity (NN)', then ['pos_-2_NNS',
        'pos_-1_IN', 'pos_0_VBG', 'pos_1_JJ', 'pos_2_NN']
        '''
        features = ['pos_0_{}'.format(self.pos)]
        features += [
                'pos_{}_{}'.format(offset, word.pos)
                for (offset, word) in self.context.items()
                if word is not None
            ]
        return features

    def get_typesixfeatures(self):
        '''Return chunk tag of words in a window length of 2

        Returns
        -------
        features : list
            A list containing the chunk tag of the current word and that of two
            words before and after the current word.

        Example
        -------

        If 'Regulating' is the current word in the context 'Cells (I-np) in
        (B-pp) Regulating (B-vp) Cellular (B-np) Immunity (I-np)', then
        ['chunk_-2_I-np', 'chunk_-1_B-pp', 'chunk_0_B-vp', 'chunk_1_B-np',
        'chunk_2_I-np'] is returned.
        '''
        features = ['chunk_0_{}'.format(self.chunk)]
        features += [
                'chunk_{}_{}'.format(offset, word.chunk)
                for (offset, word) in self.context.items()
                if word is not None
            ]
        return features

    def get_typesevenfeatures(self):
        '''Return combinations of root, chunk and part of speech tag

        Returns
        -------
        features : list
            A list containing (1) combination of root and chunk tag and part of
            speech tag of current word, (2) combination of chunk tag of current
            word and root of one word before and after the current word, and
            (3) combination of part of speech tag of current word and root of
            one word before and after the current word.

        Example
        -------

        If 'Regulating' is the current word in the context 'Cells (NNS|I-np) in
        (IN|B-pp) Regulating (VBG|B-vp) Cellular (JJ|B-np) Immunity (NN|I-np)',
        then ['R_0_regulate_C_0_B-vp', 'R_0_regulate_P_0_VBG',
        'R_-1_in_C_0_B-vp', 'R_1_cellular_C_0_B_vp', 'R_-1_in_P_0_VBG',
        'R_1_cellular_P_0_VBG']
        '''
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
            if len(line) == 0:
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
            X.append(word.get_features())
            y.append(word.get_label(binary))
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
