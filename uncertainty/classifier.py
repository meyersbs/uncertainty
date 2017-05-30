from collections import Counter
from operator import itemgetter
from os import path

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from . import constants, helpers, word
from .lib.nlp import summarizer


class Classifier(object):
    def __init__(self, granularity='word', binary=False):
        self.granularity = granularity
        self.binary = binary

        self.classifier = None
        self.vectorizer = None

    def train(self, filepath):
        if not path.exists(filepath):
            raise FileNotFoundError('No such file: {}'.format(filepath))

        lines = helpers.read_tsv(filepath)

        words = word.Words.from_lines(lines)
        X, y, _ = words.get_data(binary=self.binary)
        self.vectorizer = DictVectorizer()
        X = self.vectorizer.fit_transform(X)
        if self.binary:
            self.classifier = LogisticRegression()
        else:
            self.classifier = LogisticRegression(
                    solver='newton-cg', multi_class='multinomial'
                )
        self.classifier.fit(X, y)
        self._dump()

    def predict(self, data):
        if self.classifier is None or self.vectorizer is None:
            self._load()

        groups = None
        if type(data) is str:
            groups = summarizer.Summarizer(data).execute()
        else:
            groups = data
        words = word.Words.from_groups(groups)
        X, _, _ = words.get_data(binary=self.binary)
        X = self.vectorizer.transform(X)
        y_pred = list(self.classifier.predict(X))

        return helpers.aggregate(y_pred) if self.granularity == 'sentence' \
            else y_pred

    # Properties

    @property
    def classifier_path(self):
        return constants.BCLASS_CLASSIFIER_PATH if self.binary \
               else constants.MCLASS_CLASSIFIER_PATH

    # Private Method

    def _dump(self):
        helpers.dump(self.classifier, self.classifier_path)
        helpers.dump(self.vectorizer, constants.VECTORIZER_PATH)

    def _load(self):
        self.classifier = helpers.load(self.classifier_path)
        self.vectorizer = helpers.load(constants.VECTORIZER_PATH)
