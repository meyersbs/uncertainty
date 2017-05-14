import csv
import json
import pickle
import numpy as np
import os
import pprint
import re
#import sklearn.decomposition as decomp
import sys
import warnings

from collections import Counter
from data.merge import *
from word import *
from sentence import *

#from matplotlib.mlab import PCA
from nltk.stem.porter import *
from random import shuffle
from SuperChunker import *
from features.features import *

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
#from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore", category=FutureWarning, module="__main__")

DATA_FILE = 'data/merged_data'

BIN_CUE_MODEL = "models/binary-cue-model.p"
BIN_CUE_VECTORIZER = "vectorizers/binary-cue-vectorizer.p"
BIN_SENT_MODEL = "models/binary-sent-model.p"
BIN_SENT_VECTORIZER = "vectorizers/binary-sent-vectorizer.p"
MULTI_CUE_MODEL = "models/multiclass-cue-model.p"
MULTI_CUE_VECTORIZER = "vectorizers/multiclass-cue-vectorizer.p"
MULTI_SENT_MODEL = "models/multiclass-sent-model.p"
MULTI_SENT_VECTORIZER = "vectorizers/multiclass-sent-vectorizer.p"

PRINTER = pprint.PrettyPrinter(indent=4)
STEMMER = PorterStemmer()

################################################################################
#### HELPER FUNCTIONS ##########################################################
################################################################################

def _get_lines(filepath):
    """ Given a filepath, return a list of lines from within that file. """
    lines = None
    with open(filepath) as file:
        lines = file.readlines()
    return lines

def _get_sentences(filepath):
    """ Given a filepath, return a list of sentences from within that file. """
    sentences = list()
    _lines = list()
    for line in _get_lines(filepath):
        if line.strip() == '':  # End of Sentence
            sentences.append(_lines)
            _lines = list()
            continue
        _lines.append(line)
    return sentences

def _get_worddata(sentences, binary=True):
    """
    Given a list of sentences, return a list of sentence objects (X) and a list
    of corresponding uncertainty labels (y).
    """
    X, y = list(), list()
    for sentence in sentences:
        X_, y_, z_ = sentence.words.get_data(binary=binary)
        X.extend(X_)
        y.extend(y_)
    return X, y


################################################################################
#### CLASSIFICATION FUNCTIONS ##################################################
################################################################################

def classify(command, test_file, binary=True):
    if command == 'cue':
        words = Words.from_lines(_get_lines(test_file))
        X, y, z = words.get_data(binary=binary)

        if binary:
            vectorizer = pickle.load(open(BIN_CUE_VECTORIZER, 'rb'))
            X = vectorizer.transform(X)

            classifier = pickle.load(open(BIN_CUE_MODEL, 'rb'))
        else:
            vectorizer = pickle.load(open(MULTI_CUE_VECTORIZER, 'rb'))
            X = vectorizer.transform(X)

            classifier = pickle.load(open(MULTI_CUE_MODEL, 'rb'))

        preds = classifier.predict(X)

        _classification_report(z, preds, text="WORD:\t\t")
        return z, list(preds)
    elif command == 'sent':
        sentences = Sentences.from_lineslist(_get_sentences(test_file))
        X, y = sentences.get_data(binary=binary)

        if binary:
            vectorizer = pickle.load(open(BIN_SENT_VECTORIZER, 'rb'))
            classifier = pickle.load(open(BIN_SENT_MODEL, 'rb'))
        else:
            vectorizer = pickle.load(open(MULTI_SENT_VECTORIZER, 'rb'))
            classifier = pickle.load(open(MULTI_SENT_MODEL, 'rb'))


        preds, sents = list(), list()
        for sent in X:
            A, _, _ = sent.words.get_data(binary=binary)
            A = vectorizer.transform(A)
            cls = _classify_sentence(classifier, A, binary=binary)
            preds.append(cls[0])
            sents.append(_tag_sent(sent, cls[1]))

        _classification_report(sents, preds)
        return sents, preds

def _tag_sent(sent, labels):
    sent = sent.get_sentence().split()

    tagged_sent = []
    for word, label in zip(sent, labels):
        if label == "C":
            tagged_sent.append(word)
        else:
            tagged_sent.append("(" + word + "-" + label + ")")

    return " ".join(tagged_sent)

def _classify_sentence(classifier, X, binary=True):
    y_pred = classifier.predict(X)
    #print(y_pred)
    #y_pred = classifier.predict_proba(X)

    if binary:
        for label in y_pred:
            if label != "C":
                return "U", y_pred
        return "C", y_pred
    else:
        labs = {"C": 0, "U": 0, "I": 0, "N": 0, "E": 0, "D": 0}
        for label in y_pred:
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
                return max_keys[0], y_pred
            else:
                return "U", y_pred
        else:
            return "C", y_pred
'''
def _pca(X, y):
    data = np.array(X, y)
    results = PCA(data)
    print(results.fracs)
    input()
    print(results.Y)
    input()
    return results

def _pca(X, y):
    data = (X - np.mean(X, 0)) / np.std(X, 0)
    pca = decomp.PCA(n_components=30)
    results = pca.fit_transform(data)
    print(results)
    input()
    return results
'''

################################################################################
#### TRAINING FUNCTIONS ########################################################
################################################################################

def cue(data=DATA_FILE, binary=True):
    print("Gathering Documents...")
    words = Words.from_lines(_get_lines(data))
    X, y, _ = words.get_data(binary=binary)

    print("Splitting Train/Test Groups...")
    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, stratify=y
        )

    print("Generating Feature Vectors...")
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    print("Training Classifier...")
    classifier = object()
    if binary:
        classifier = LogisticRegression(n_jobs=-1)
    else:
        classifier = LogisticRegression(solver='newton-cg',
                                        multi_class='multinomial', n_jobs=-1)

    print("Fitting Model...")
    classifier.fit(X_train, y_train)

    print("Running Predictions...")
    y_pred = classifier.predict(X_test)

    _show_performance(y_test, y_pred, binary=binary)

    if binary:
        print("Dumping Classifier to Disk...")
        pickle.dump(classifier, open(BIN_CUE_MODEL, 'wb'))
        print("Dumping Vectorizer to Disk...")
        pickle.dump(vectorizer, open(BIN_CUE_VECTORIZER, 'wb'))
    else:
        print("Dumping Classifier to Disk...")
        pickle.dump(classifier, open(MULTI_CUE_MODEL, 'wb'))
        print("Dumping Vectorizer to Disk...")
        pickle.dump(vectorizer, open(MULTI_CUE_VECTORIZER, 'wb'))

    print("Cleaning Up...")

def sentence(data=DATA_FILE, binary=True):
    print("Gathering Documents...")
    sentences = Sentences.from_lineslist(_get_sentences(data))
    X, y = sentences.get_data(binary=binary)

    print("Splitting Train/Test Groups...")
    s_train, s_test, g_train, g_test = train_test_split(
                X, y, test_size=0.25, stratify=y
        )

    print("Parsing Words...")
    X_train, y_train = _get_worddata(s_train, binary=binary)

    print("Generating Feature Vectors...")
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(X_train)

    print("Training Classifier...")
    classifier = object()
    if binary:
        classifier = LogisticRegression(n_jobs=-1)
    else:
        classifier = LogisticRegression(solver='newton-cg',
                                        multi_class='multinomial', n_jobs=-1)

    print("Fitting Model...")
    classifier.fit(X_train, y_train)
    #classifier.fit_transform(X_train, y_train)

    print("Running Predictions...")
    y_pred = list()
    for sentence in s_test:
        X_test, _, _ = sentence.words.get_data(binary=binary)
        X_test = vectorizer.transform(X_test)

        y_pred.append(_classify_sentence(classifier, X_test, binary=binary)[0])
    _show_performance(g_test, y_pred, binary=binary)

    if binary:
        print("Dumping Classifier to Disk...")
        pickle.dump(classifier, open(BIN_SENT_MODEL, 'wb'))
        print("Dumping Vectorizer to Disk...")
        pickle.dump(vectorizer, open(BIN_SENT_VECTORIZER, 'wb'))
    else:
        print("Dumping Classifier to Disk...")
        pickle.dump(classifier, open(MULTI_SENT_MODEL, 'wb'))
        print("Dumping Vectorizer to Disk...")
        pickle.dump(vectorizer, open(MULTI_SENT_VECTORIZER, 'wb'))

    print("Cleaning Up...")

################################################################################
#### PREPROCESSING FUNCTIONS ###################################################
################################################################################

def features(data_file):
    """
    This preprocessing function takes in a data_file containing one sentence
    per line, generates the features for each token, and writes them to a file
    with the same name and '.tsv' appended to the end. See the documents
    /test_data.txt and /test_data.txt.tsv for examples of valid input and
    output, respectively.

    This preprocessing step allows for the features of a corpus to be generated
    only once, greatly speeding up the classification process.

    NOTE: The output file will be much larger in size than the input file.
    """
    with open(data_file, 'r') as f:
        filename = data_file + '.tsv'
        with open(filename, 'w', newline='') as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter='\t', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
            for i, line in enumerate(f.readlines()):
                if line.strip() == '':
                    tsv_writer.writerow('')
                else:
                    features = get_features(line)
                    for key, val in sorted(features.items()):
                        (tok_num, tok) = key.split("_")
                        row = ['sent' + str(i) + "token" + str(tok_num),
                               str(tok), str(STEMMER.stem(tok)).lower(),
                               pos_tag(tok)[0][1], 'X', 'X']
                        for k, v in sorted(val.items()):
                            row.append(str(k) + ":" + str(v))
                        tsv_writer.writerow(row)

def merge(json_data, tagged_data):
    """ Passthrough function. See /data/merge.py:merge_data() for details. """
    merge_data(json_data, tagged_data)

################################################################################
#### OUTPUT FORMATTING #########################################################
################################################################################

def _help():
    """ Display help text to the user. """
    sys.stderr.write("USAGE: {} command\n".format(sys.argv[0]))
    sys.stderr.write("  command: One of {}".format(",".join(COMMANDS.keys())))
    sys.stderr.write("\n")
    sys.exit(1)

def _classification_report(elems, preds, text="SENTENCE:\t"):
    """ Display a pretty, custom classification report. """
    categories = {"C": "Certain", "U": "Uncertain", "E": "Epistemic",
                  "D": "Doxastic", "I": "Investigation", "N": "Condition"}
    for i, elem in enumerate(elems):
        #print(text + elem)
        try:
            print("  [" + categories[preds[i]] + "]\t" + elem)
        except Exception as e:
            print("  [ERROR]\t" + elem)
            continue

    return True

def _show_performance(y_test, y_pred, binary=True):
    """ Display Precision, Recall, and F1 for the target label(s). """
    if binary:
        print("####### Classification Report ###########")
        print()
        print(classification_report(y_test, y_pred))
        print()
        print("####### Confusion Matrix ################")
        print()
        print(confusion_matrix(y_test, y_pred, labels=["C", "U"]))
    else:
        multi_labels = ["C", "U", "E", "D", "I", "N"]
        print("####### Classification Report ###########")
        print()
        print(classification_report(y_test, y_pred))
        print()
        print("####### Confusion Matrix ################")
        print()
        print(confusion_matrix(y_test, y_pred, labels=multi_labels))

################################################################################
#### INPUT HANDLING ############################################################
################################################################################

def _is_valid(flag):
    return False if flag == '-m' else True

def _exists(filename):
    return os.path.exists(filename)

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 2 or len(args) > 4:
        _help()
    elif len(args) == 2 and args[0] == 'cue':
        cue(binary=_is_valid(args[1]))
    elif len(args) == 2 and args[0] == 'sent':
        sentence(binary=_is_valid(args[1]))
    elif len(args) == 2 and args[0] == 'features' and _exists(args[1]):
        features(args[1])
    elif (len(args) == 3 and args[0] == 'merge' and _exists(args[1])
          and _exists(args[2])):
        merge(args[1], args[2])
    elif (len(args) == 4 and args[0] == 'classify'
          and args[1] in ['sent', 'cue'] and _exists(args[2])):
        classify(args[1], args[2], _is_valid(args[3]))
    else:
        _help()
