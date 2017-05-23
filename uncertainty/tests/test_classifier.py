import unittest

from uncertainty.classifier import Classifier


class TestClassifierWordBinary(unittest.TestCase):
    def setUp(self):
        self.classifier = Classifier(granularity='word', binary=True)

    def test_predict_for_groups(self):
        data = [
                ('Its', 'it', 'PRP$', 'B-np'),
                ('short', 'short', 'JJ', 'I-np'),
                ('life', 'life', 'NN', 'I-np'),
                ('span', 'span', 'NN', 'I-np'),
                ('(', '', '-LRB-', 'O'),
                ('thirty-odd', 'thirtyodd', 'JJ', 'B-np'),
                ('episodes', 'episod', 'NNS', 'I-np'),
                (')', '', '-RRB-', 'O'),
                ('was', 'wa', 'VBD', 'B-vp'),
                ('in', 'in', 'IN', 'B-pp'),
                ('part', 'part', 'NN', 'B-np'),
                ('because', 'becaus', 'IN', 'B-sbar'),
                ('it', 'it', 'PRP', 'B-np'),
                ('was', 'wa', 'VBD', 'B-vp'),
                ('considered', 'consid', 'VBN', 'I-vp'),
                ('too', 'too', 'SO', 'B-adjp'),
                ('violent', 'violent', 'JJ', 'I-adjp'),
                ('at', 'at', 'IN', 'B-pp'),
                ('the', 'the', 'DT', 'B-np'),
                ('time', 'time', 'NN', 'I-np'),
                (',', '', ',', 'O'),
                ('although', 'although', 'IN', 'B-sbar'),
                ('the', 'the', 'DT', 'B-np'),
                ('violence', 'violenc', 'NN', 'I-np'),
                ('it', 'it', 'PRP', 'O'),
                ('depicted', 'depict', 'VBN', 'B-vp'),
                ('is', 'is', 'VBZ', 'O'),
                ('actually', 'actual', 'RB', 'B-advp'),
                ('mild', 'mild', 'JJ', 'B-np'),
                ('by', 'by', 'IN', 'B-pp'),
                ('today\'s', 'todai', 'NNS', 'B-np'),
                ('television', 'televi', 'NN', 'I-np'),
                ('standards', 'standard', 'NNS', 'I-np'),
                ('.', '', '.', 'O'),
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
                'C',  # television
                'C',  # standards
                'C'   # .
            ]
        actual = self.classifier.predict(data)

        self.assertEqual(expected, actual)

    def test_predict_for_sentence(self):
        with self.assertRaises(NotImplementedError):
            self.classifier.predict('I am certain.')


class TestClassifierSentenceBinary(unittest.TestCase):
    def setUp(self):
        self.classifier = Classifier(granularity='sentence', binary=True)

    def test_predict_for_groups(self):
        data = [
                ('Its', 'it', 'PRP$', 'B-np'),
                ('short', 'short', 'JJ', 'I-np'),
                ('life', 'life', 'NN', 'I-np'),
                ('span', 'span', 'NN', 'I-np'),
                ('(', '', '-LRB-', 'O'),
                ('thirty-odd', 'thirtyodd', 'JJ', 'B-np'),
                ('episodes', 'episod', 'NNS', 'I-np'),
                (')', '', '-RRB-', 'O'),
                ('was', 'wa', 'VBD', 'B-vp'),
                ('in', 'in', 'IN', 'B-pp'),
                ('part', 'part', 'NN', 'B-np'),
                ('because', 'becaus', 'IN', 'B-sbar'),
                ('it', 'it', 'PRP', 'B-np'),
                ('was', 'wa', 'VBD', 'B-vp'),
                ('considered', 'consid', 'VBN', 'I-vp'),
                ('too', 'too', 'SO', 'B-adjp'),
                ('violent', 'violent', 'JJ', 'I-adjp'),
                ('at', 'at', 'IN', 'B-pp'),
                ('the', 'the', 'DT', 'B-np'),
                ('time', 'time', 'NN', 'I-np'),
                (',', '', ',', 'O'),
                ('although', 'although', 'IN', 'B-sbar'),
                ('the', 'the', 'DT', 'B-np'),
                ('violence', 'violenc', 'NN', 'I-np'),
                ('it', 'it', 'PRP', 'O'),
                ('depicted', 'depict', 'VBN', 'B-vp'),
                ('is', 'is', 'VBZ', 'O'),
                ('actually', 'actual', 'RB', 'B-advp'),
                ('mild', 'mild', 'JJ', 'B-np'),
                ('by', 'by', 'IN', 'B-pp'),
                ('today\'s', 'todai', 'NNS', 'B-np'),
                ('television', 'televi', 'NN', 'I-np'),
                ('standards', 'standard', 'NNS', 'I-np'),
                ('.', '', '.', 'O'),
            ]

        expected = 'U'
        actual = self.classifier.predict(data)
        self.assertEqual(expected, actual)


class TestClassifierWordMulti(unittest.TestCase):
    def setUp(self):
        self.classifier = Classifier(granularity='word', binary=False)

    def test_predict_for_groups(self):
        data = [
                ('Its', 'it', 'PRP$', 'B-np'),
                ('short', 'short', 'JJ', 'I-np'),
                ('life', 'life', 'NN', 'I-np'),
                ('span', 'span', 'NN', 'I-np'),
                ('(', '', '-LRB-', 'O'),
                ('thirty-odd', 'thirtyodd', 'JJ', 'B-np'),
                ('episodes', 'episod', 'NNS', 'I-np'),
                (')', '', '-RRB-', 'O'),
                ('was', 'wa', 'VBD', 'B-vp'),
                ('in', 'in', 'IN', 'B-pp'),
                ('part', 'part', 'NN', 'B-np'),
                ('because', 'becaus', 'IN', 'B-sbar'),
                ('it', 'it', 'PRP', 'B-np'),
                ('was', 'wa', 'VBD', 'B-vp'),
                ('considered', 'consid', 'VBN', 'I-vp'),
                ('too', 'too', 'SO', 'B-adjp'),
                ('violent', 'violent', 'JJ', 'I-adjp'),
                ('at', 'at', 'IN', 'B-pp'),
                ('the', 'the', 'DT', 'B-np'),
                ('time', 'time', 'NN', 'I-np'),
                (',', '', ',', 'O'),
                ('although', 'although', 'IN', 'B-sbar'),
                ('the', 'the', 'DT', 'B-np'),
                ('violence', 'violenc', 'NN', 'I-np'),
                ('it', 'it', 'PRP', 'O'),
                ('depicted', 'depict', 'VBN', 'B-vp'),
                ('is', 'is', 'VBZ', 'O'),
                ('actually', 'actual', 'RB', 'B-advp'),
                ('mild', 'mild', 'JJ', 'B-np'),
                ('by', 'by', 'IN', 'B-pp'),
                ('today\'s', 'todai', 'NNS', 'B-np'),
                ('television', 'televi', 'NN', 'I-np'),
                ('standards', 'standard', 'NNS', 'I-np'),
                ('.', '', '.', 'O'),
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
                'C',  # television
                'C',  # standards
                'C'   # .
            ]
        actual = self.classifier.predict(data)

        self.assertEqual(expected, actual)


class TestClassifierSentenceMulti(unittest.TestCase):
    def setUp(self):
        self.classifier = Classifier(granularity='sentence', binary=False)

    def test_predict_for_groups(self):
        data = [
                ('Its', 'it', 'PRP$', 'B-np'),
                ('short', 'short', 'JJ', 'I-np'),
                ('life', 'life', 'NN', 'I-np'),
                ('span', 'span', 'NN', 'I-np'),
                ('(', '', '-LRB-', 'O'),
                ('thirty-odd', 'thirtyodd', 'JJ', 'B-np'),
                ('episodes', 'episod', 'NNS', 'I-np'),
                (')', '', '-RRB-', 'O'),
                ('was', 'wa', 'VBD', 'B-vp'),
                ('in', 'in', 'IN', 'B-pp'),
                ('part', 'part', 'NN', 'B-np'),
                ('because', 'becaus', 'IN', 'B-sbar'),
                ('it', 'it', 'PRP', 'B-np'),
                ('was', 'wa', 'VBD', 'B-vp'),
                ('considered', 'consid', 'VBN', 'I-vp'),
                ('too', 'too', 'SO', 'B-adjp'),
                ('violent', 'violent', 'JJ', 'I-adjp'),
                ('at', 'at', 'IN', 'B-pp'),
                ('the', 'the', 'DT', 'B-np'),
                ('time', 'time', 'NN', 'I-np'),
                (',', '', ',', 'O'),
                ('although', 'although', 'IN', 'B-sbar'),
                ('the', 'the', 'DT', 'B-np'),
                ('violence', 'violenc', 'NN', 'I-np'),
                ('it', 'it', 'PRP', 'O'),
                ('depicted', 'depict', 'VBN', 'B-vp'),
                ('is', 'is', 'VBZ', 'O'),
                ('actually', 'actual', 'RB', 'B-advp'),
                ('mild', 'mild', 'JJ', 'B-np'),
                ('by', 'by', 'IN', 'B-pp'),
                ('today\'s', 'todai', 'NNS', 'B-np'),
                ('television', 'televi', 'NN', 'I-np'),
                ('standards', 'standard', 'NNS', 'I-np'),
                ('.', '', '.', 'O'),
            ]

        expected = 'D'
        actual = self.classifier.predict(data)

        self.assertEqual(expected, actual)
