import unittest

from uncertainty.classifier import Classifier


class TestClassifierWordBinary(unittest.TestCase):
    def setUp(self):
        self.classifier = Classifier(granularity='word', binary=True)

    def test_predict_for_groups(self):
        data = [
                ('Its', 'it', 'PRP$', 'B-NP'),
                ('short', 'short', 'JJ', 'I-NP'),
                ('life', 'life', 'NN', 'I-NP'),
                ('span', 'span', 'NN', 'I-NP'),
                ('(', '(', '(', 'O'),
                ('thirty-odd', 'thirty-odd', 'JJ', 'B-NP'),
                ('episodes', 'episod', 'NNS', 'I-NP'),
                (')', ')', ')', 'O'),
                ('was', 'wa', 'VBD', 'B-VP'),
                ('in', 'in', 'IN', 'B-PP'),
                ('part', 'part', 'NN', 'B-NP'),
                ('because', 'becaus', 'IN', 'B-PP'),
                ('it', 'it', 'PRP', 'B-NP'),
                ('was', 'wa', 'VBD', 'B-VP'),
                ('considered', 'consid', 'VBN', 'I-VP'),
                ('too', 'too', 'RB', 'O'),
                ('violent', 'violent', 'JJ', 'B-NP'),
                ('at', 'at', 'IN', 'B-PP'),
                ('the', 'the', 'DT', 'B-NP'),
                ('time', 'time', 'NN', 'I-NP'),
                (',', ',', ',', 'O'),
                ('although', 'although', 'IN', 'O'),
                ('the', 'the', 'DT', 'B-NP'),
                ('violence', 'violenc', 'NN', 'I-NP'),
                ('it', 'it', 'PRP', 'B-NP'),
                ('depicted', 'depict', 'VBD', 'B-VP'),
                ('is', 'is', 'VBZ', 'I-VP'),
                ('actually', 'actual', 'RB', 'O'),
                ('mild', 'mild', 'VBN', 'O'),
                ('by', 'by', 'IN', 'O'),
                ('today', 'today', 'NN', 'B-NP'),
                ("'s", "'s", 'POS', 'B-NP'),
                ('television', 'televis', 'NN', 'I-NP'),
                ('standards', 'standard', 'NNS', 'I-NP'),
                ('.', '.', '.', 'O')
            ]
        expected = [
                'C',  # Its
                'C',  # short
                'C',  # life
                'C',  # span
                'C',  # (
                'C',  # thirty-odd
                'C',  # episodes
                'C',  # )
                'C',  # was
                'C',  # in
                'C',  # part
                'C',  # because
                'C',  # it
                'C',  # was
                'U',  # considered
                'C',  # too
                'C',  # violent
                'C',  # at
                'C',  # the
                'C',  # time
                'C',  # ,
                'C',  # although
                'C',  # the
                'C',  # violence
                'C',  # it
                'C',  # depicted
                'C',  # is
                'C',  # actually
                'C',  # mild
                'C',  # by
                'C',  # today
                'C',  # 's
                'C',  # television
                'C',  # standards
                'C'   # .
            ]
        actual = self.classifier.predict(data)
        self.assertEqual(expected, actual)

    def test_predict_for_sentence(self):
        data = 'Its short life span (thirty-odd episodes) was in part ' \
               'because  it was considered too violent at the time, ' \
               'although the violence it depicted is actually mild by ' \
               'today\'s television standards.'
        expected = [
                'C',  # Its
                'C',  # short
                'C',  # life
                'C',  # span
                'C',  # (
                'C',  # thirty-odd
                'C',  # episodes
                'C',  # )
                'C',  # was
                'C',  # in
                'C',  # part
                'C',  # because
                'C',  # it
                'C',  # was
                'U',  # considered
                'C',  # too
                'C',  # violent
                'C',  # at
                'C',  # the
                'C',  # time
                'C',  # ,
                'C',  # although
                'C',  # the
                'C',  # violence
                'C',  # it
                'C',  # depicted
                'C',  # is
                'C',  # actually
                'C',  # mild
                'C',  # by
                'C',  # today
                'C',  # 's
                'C',  # television
                'C',  # standards
                'C'   # .
            ]
        actual = self.classifier.predict(data)
        self.assertEqual(expected, actual)


class TestClassifierSentenceBinary(unittest.TestCase):
    def setUp(self):
        self.classifier = Classifier(granularity='sentence', binary=True)

    def test_predict_for_groups(self):
        data = [
                ('Its', 'it', 'PRP$', 'B-NP'),
                ('short', 'short', 'JJ', 'I-NP'),
                ('life', 'life', 'NN', 'I-NP'),
                ('span', 'span', 'NN', 'I-NP'),
                ('(', '(', '(', 'O'),
                ('thirty-odd', 'thirty-odd', 'JJ', 'B-NP'),
                ('episodes', 'episod', 'NNS', 'I-NP'),
                (')', ')', ')', 'O'),
                ('was', 'wa', 'VBD', 'B-VP'),
                ('in', 'in', 'IN', 'B-PP'),
                ('part', 'part', 'NN', 'B-NP'),
                ('because', 'becaus', 'IN', 'B-PP'),
                ('it', 'it', 'PRP', 'B-NP'),
                ('was', 'wa', 'VBD', 'B-VP'),
                ('considered', 'consid', 'VBN', 'I-VP'),
                ('too', 'too', 'RB', 'O'),
                ('violent', 'violent', 'JJ', 'B-NP'),
                ('at', 'at', 'IN', 'B-PP'),
                ('the', 'the', 'DT', 'B-NP'),
                ('time', 'time', 'NN', 'I-NP'),
                (',', ',', ',', 'O'),
                ('although', 'although', 'IN', 'O'),
                ('the', 'the', 'DT', 'B-NP'),
                ('violence', 'violenc', 'NN', 'I-NP'),
                ('it', 'it', 'PRP', 'B-NP'),
                ('depicted', 'depict', 'VBD', 'B-VP'),
                ('is', 'is', 'VBZ', 'I-VP'),
                ('actually', 'actual', 'RB', 'O'),
                ('mild', 'mild', 'VBN', 'O'),
                ('by', 'by', 'IN', 'O'),
                ('today', 'today', 'NN', 'B-NP'),
                ("'s", "'s", 'POS', 'B-NP'),
                ('television', 'televis', 'NN', 'I-NP'),
                ('standards', 'standard', 'NNS', 'I-NP'),
                ('.', '.', '.', 'O')
            ]
        expected = 'U'
        actual = self.classifier.predict(data)
        self.assertEqual(expected, actual)

    def test_predict_for_sentence(self):
        data = 'Its short life span (thirty-odd episodes) was in part ' \
               'because  it was considered too violent at the time, ' \
               'although the violence it depicted is actually mild by ' \
               'today\'s television standards.'

        expected = 'U'
        actual = self.classifier.predict(data)
        self.assertEqual(expected, actual)


class TestClassifierWordMulti(unittest.TestCase):
    def setUp(self):
        self.classifier = Classifier(granularity='word', binary=False)

    def test_predict_for_groups(self):
        data = [
                ('Its', 'it', 'PRP$', 'B-NP'),
                ('short', 'short', 'JJ', 'I-NP'),
                ('life', 'life', 'NN', 'I-NP'),
                ('span', 'span', 'NN', 'I-NP'),
                ('(', '(', '(', 'O'),
                ('thirty-odd', 'thirty-odd', 'JJ', 'B-NP'),
                ('episodes', 'episod', 'NNS', 'I-NP'),
                (')', ')', ')', 'O'),
                ('was', 'wa', 'VBD', 'B-VP'),
                ('in', 'in', 'IN', 'B-PP'),
                ('part', 'part', 'NN', 'B-NP'),
                ('because', 'becaus', 'IN', 'B-PP'),
                ('it', 'it', 'PRP', 'B-NP'),
                ('was', 'wa', 'VBD', 'B-VP'),
                ('considered', 'consid', 'VBN', 'I-VP'),
                ('too', 'too', 'RB', 'O'),
                ('violent', 'violent', 'JJ', 'B-NP'),
                ('at', 'at', 'IN', 'B-PP'),
                ('the', 'the', 'DT', 'B-NP'),
                ('time', 'time', 'NN', 'I-NP'),
                (',', ',', ',', 'O'),
                ('although', 'although', 'IN', 'O'),
                ('the', 'the', 'DT', 'B-NP'),
                ('violence', 'violenc', 'NN', 'I-NP'),
                ('it', 'it', 'PRP', 'B-NP'),
                ('depicted', 'depict', 'VBD', 'B-VP'),
                ('is', 'is', 'VBZ', 'I-VP'),
                ('actually', 'actual', 'RB', 'O'),
                ('mild', 'mild', 'VBN', 'O'),
                ('by', 'by', 'IN', 'O'),
                ('today', 'today', 'NN', 'B-NP'),
                ("'s", "'s", 'POS', 'B-NP'),
                ('television', 'televis', 'NN', 'I-NP'),
                ('standards', 'standard', 'NNS', 'I-NP'),
                ('.', '.', '.', 'O')
            ]
        expected = [
                'C',  # Its
                'C',  # short
                'C',  # life
                'C',  # span
                'C',  # (
                'C',  # thirty-odd
                'C',  # episodes
                'C',  # )
                'C',  # was
                'C',  # in
                'C',  # part
                'C',  # because
                'C',  # it
                'C',  # was
                'D',  # considered
                'C',  # too
                'C',  # violent
                'C',  # at
                'C',  # the
                'C',  # time
                'C',  # ,
                'C',  # although
                'C',  # the
                'C',  # violence
                'C',  # it
                'C',  # depicted
                'C',  # is
                'C',  # actually
                'C',  # mild
                'C',  # by
                'C',  # today
                'C',  # 's
                'C',  # television
                'C',  # standards
                'C'   # .
            ]
        actual = self.classifier.predict(data)
        self.assertEqual(expected, actual)

    def test_predict_for_sentence(self):
        data = 'Its short life span (thirty-odd episodes) was in part ' \
               'because  it was considered too violent at the time, ' \
               'although the violence it depicted is actually mild by ' \
               'today\'s television standards.'
        expected = [
                'C',  # Its
                'C',  # short
                'C',  # life
                'C',  # span
                'C',  # (
                'C',  # thirty-odd
                'C',  # episodes
                'C',  # )
                'C',  # was
                'C',  # in
                'C',  # part
                'C',  # because
                'C',  # it
                'C',  # was
                'D',  # considered
                'C',  # too
                'C',  # violent
                'C',  # at
                'C',  # the
                'C',  # time
                'C',  # ,
                'C',  # although
                'C',  # the
                'C',  # violence
                'C',  # it
                'C',  # depicted
                'C',  # is
                'C',  # actually
                'C',  # mild
                'C',  # by
                'C',  # today
                'C',  # 's
                'C',  # television
                'C',  # standards
                'C'   # .
            ]
        actual = self.classifier.predict(data)
        self.assertEqual(expected, actual)

        data = 'If I\'m human, then this test will pass without failure.'
        expected = [
                'N',  # If
                'C',  # I
                'C',  # 'm
                'C',  # human
                'C',  # ,
                'C',  # then
                'C',  # this
                'C',  # test
                'C',  # will
                'C',  # pass
                'C',  # without
                'C',  # failure
                'C',  # .
            ]
        actual = self.classifier.predict(data)
        self.assertEqual(expected, actual)

        data = 'I am certain that this sentence will be certain.'

        expected = [
                'C',  # I
                'C',  # am
                'C',  # certain
                'C',  # that
                'C',  # this
                'C',  # sentence
                'C',  # will
                'C',  # be
                'C',  # certain
                'C',  # .
            ]
        actual = self.classifier.predict(data)
        self.assertEqual(expected, actual)


class TestClassifierSentenceMulti(unittest.TestCase):
    def setUp(self):
        self.classifier = Classifier(granularity='sentence', binary=False)

    def test_predict_for_groups(self):
        data = [
                ('Its', 'it', 'PRP$', 'B-NP'),
                ('short', 'short', 'JJ', 'I-NP'),
                ('life', 'life', 'NN', 'I-NP'),
                ('span', 'span', 'NN', 'I-NP'),
                ('(', '(', '(', 'O'),
                ('thirty-odd', 'thirty-odd', 'JJ', 'B-NP'),
                ('episodes', 'episod', 'NNS', 'I-NP'),
                (')', ')', ')', 'O'),
                ('was', 'wa', 'VBD', 'B-VP'),
                ('in', 'in', 'IN', 'B-PP'),
                ('part', 'part', 'NN', 'B-NP'),
                ('because', 'becaus', 'IN', 'B-PP'),
                ('it', 'it', 'PRP', 'B-NP'),
                ('was', 'wa', 'VBD', 'B-VP'),
                ('considered', 'consid', 'VBN', 'I-VP'),
                ('too', 'too', 'RB', 'O'),
                ('violent', 'violent', 'JJ', 'B-NP'),
                ('at', 'at', 'IN', 'B-PP'),
                ('the', 'the', 'DT', 'B-NP'),
                ('time', 'time', 'NN', 'I-NP'),
                (',', ',', ',', 'O'),
                ('although', 'although', 'IN', 'O'),
                ('the', 'the', 'DT', 'B-NP'),
                ('violence', 'violenc', 'NN', 'I-NP'),
                ('it', 'it', 'PRP', 'B-NP'),
                ('depicted', 'depict', 'VBD', 'B-VP'),
                ('is', 'is', 'VBZ', 'I-VP'),
                ('actually', 'actual', 'RB', 'O'),
                ('mild', 'mild', 'VBN', 'O'),
                ('by', 'by', 'IN', 'O'),
                ('today', 'today', 'NN', 'B-NP'),
                ("'s", "'s", 'POS', 'B-NP'),
                ('television', 'televis', 'NN', 'I-NP'),
                ('standards', 'standard', 'NNS', 'I-NP'),
                ('.', '.', '.', 'O')
            ]

        expected = 'D'
        actual = self.classifier.predict(data)

        self.assertEqual(expected, actual)

    def test_predict_for_sentence(self):
        data = 'Its short life span (thirty-odd episodes) was in part ' \
               'because  it was considered too violent at the time, ' \
               'although the violence it depicted is actually mild by ' \
               'today\'s television standards.'

        expected = 'D'
        actual = self.classifier.predict(data)
        self.assertEqual(expected, actual)

        data = 'If I\'m human, then this test will pass without failure.'

        expected = 'N'
        actual = self.classifier.predict(data)
        self.assertEqual(expected, actual)

        data = 'I am certain that this sentence will be certain.'

        expected = 'C'
        actual = self.classifier.predict(data)
        self.assertEqual(expected, actual)
