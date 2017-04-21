import csv
import _pickle
import pprint
import sys

from word import *
from sentence import *
from nltk.stem.porter import *
from SuperChunker import *
from features.features import *

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
        classification_report, precision_recall_fscore_support
    )
from sklearn.model_selection import train_test_split

DATA_FILE = 'data/merged_data'
PRINTER = pprint.PrettyPrinter(indent=4, width=80)
STEMMER = PorterStemmer()

def classify(command, test_file):
    if command == 'cue':
        words = Words(_get_lines(test_file))
        X, y = words.get_data()
        vectorizer = DictVectorizer()

        classifier = _pickle.load(open('uncertainty-cue-model.p', 'rb'))
        y_pred = classifier.predict(X)
        _show_performance(y, y_pred)
    elif command == 'sentence':
        sentences = Sentences(_get_sentences(test_file))
        X, y = sentences.get_data()
        S, _, g, _ = train_test_split(X, y, test_size=0.0)
        X, y = _get_worddata(S)

        vectorizer = DictVectorizer()
        classifier = _pickle.load(open('uncertainty-sent-model.p', 'rb'))

        y_pred = list()
        for sent in S:
            A, _ = sent.words.get_data()
            A = vectorizer.transform(A)
            y_pred.append(_classify_sentence(classifier, A))

        _show_performance(g, y_pred)

def cue(data=DATA_FILE):
    words = Words(_get_lines(data))
    X, y = words.get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    classifier = LogisticRegression(n_jobs=-1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    _show_performance(y_test, y_pred)

    _pickle.dump(classifier, open('uncertainty-cue-model.p', 'wb'))


def sentence(data=DATA_FILE):
    sentences = Sentences(_get_sentences(data))
    X, y = sentences.get_data()

    s_train, s_test, g_train, g_test = train_test_split(X, y, test_size=0.25)

    X_train, y_train = _get_worddata(s_train)

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(X_train)

    classifier = LogisticRegression(n_jobs=-1)
    classifier.fit(X_train, y_train)

    y_pred = list()
    for sentence in s_test:
        X_test, _ = sentence.words.get_data()
        X_test = vectorizer.transform(X_test)
        y_pred.append(_classify_sentence(classifier, X_test))

    _show_performance(g_test, y_pred)

    _pickle.dump(classifier, open('uncertainty-sent-model.p', 'wb'))

def features(data_file):
    with open(data_file, 'r') as f:
        filename = data_file + '.tsv'
        with open(filename, 'w', newline='') as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter='\t', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
            for i, line in enumerate(f.readlines()):
                features = get_features(line)
                #print("=====")
                #PRINTER.pprint(features)
                for key, val in sorted(features.items()):
                    (tok_num, tok) = key.split("_")
                    row = ['sent' + str(i) + "token" + str(tok_num), str(tok),
                           str(STEMMER.stem(tok)).lower(), 'X']
                    for k, v in sorted(val.items()):
                        row.append(str(k) + ":" + str(v))
                    tsv_writer.writerow(row)

def _classify_sentence(classifier, X):
    y_pred = classifier.predict(X)
    for item in y_pred:
        if item == 'u':
            return item
    return 'c'


def _get_lines(filepath):
    lines = None
    with open(filepath) as file:
        lines = file.readlines()
    return lines


def _get_sentences(filepath):
    sentences = list()
    _lines = list()
    for line in _get_lines(filepath):
        if line.strip() == '':  # End of Sentence
            sentences.append(_lines)
            _lines = list()
            continue
        _lines.append(line)
    return sentences


def _get_worddata(sentences):
    X, y = list(), list()
    for sentence in sentences:
        X_, y_ = sentence.words.get_data()
        X.extend(X_)
        y.extend(y_)
    return X, y


def _help():
    sys.stderr.write('USAGE: {} command\n'.format(sys.argv[0]))
    sys.stderr.write('  command: One of {}\n'.format(
            ','.join(COMMANDS.keys())
        ))
    sys.exit(1)


def _show_performance(y_test, y_pred):
    precision, recall, fscore, support = precision_recall_fscore_support(
            y_test, y_pred, average='binary', pos_label='u'
        )
    print('{} Performance'.format('#' * 7))
    print()
    print('  {:15} {:3.2%}\n  {:15} {:3.2%}\n  {:15} {:3.2%}'.format(
            'Precision', precision, 'Recall', recall, 'F1', fscore
        ))
    print()
    print('{} Classification Report'.format('#' * 7))
    print()
    print(classification_report(y_test, y_pred))

COMMANDS = {'cue': cue, 'sentence': sentence, 'classify': classify, 'features': features}

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        _help()
    elif len(args) == 1 and args[0] in COMMANDS:
        COMMANDS[args[0]]()
    elif len(args) == 2:
        COMMANDS[args[0]](args[1])
    elif len(args) == 3 and args[0] in COMMANDS and args[1] in COMMANDS:
        COMMANDS[args[0]](args[1], args[2])
    else:
        _help()
