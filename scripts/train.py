import _pickle
import json
import numpy as np
import os
import random
import sys

from sklearn import svm
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report, confusion_matrix

from features import *
from features.features import *

def train_svm(documents, ntesting=500):

    random.shuffle(documents)
    testing = documents[-ntesting:]
    training = documents[:-ntesting]

    print("Generating feature vectors for the training data...")
    x, y = docs_to_feature_vectors(training)
    print("Generating feature vectors for the testing data...")
    xtest, ytest = docs_to_feature_vectors(testing)

    print("Training the classifier...")
    classifier = svm.SVC(C=0.02, kernel='linear', probability=True)
    print("Fitting the classifier...")
    classifier.fit(x, y)

    print("Running against testing data...")
    y_prediction = classifier.predict(xtest)
#    print(xtest)
#    print(ytest)
    print(classification_report(ytest, y_prediction))
    print("\n")
    print(confusion_matrix(ytest, y_prediction))

    return classifier


def docs_to_feature_vectors(documents):
    feature_keys = False
    x, y = [], []
    cnt = 0

    for doc in documents:
        feature_set = get_features(doc)
        if not feature_keys:
            feature_keys = sorted(feature_set.keys())

        feature_vector = [feature_set[feature] for feature in feature_keys]

        doc_class = 0 # label: not_uncertain
        if len(doc['ccue']) > 0:
            doc_class = 1 # label: uncertain

#        print(doc_class)
        x.append(feature_vector)
        y.append(doc_class)
        print(cnt)
        cnt+=1

    x = csr_matrix(np.asarray(x))
    y = np.asarray(y)

#    print(x)
#    print(y)
    return x, y
