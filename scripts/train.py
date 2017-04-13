import _pickle
import json
import numpy as np
import os
import random
import sys

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.metrics import (
        classification_report, confusion_matrix,
        precision_recall_fscore_support
    )

from features import *
from features.features import *


def train(documents, test_size=0.25):
    X, y = docs_to_feature_vectors(documents)
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size
        )

    classifier = linear_model.LogisticRegression(
            class_weight="balanced", n_jobs=-1
        )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print("Performance")
    print("-----------")
    (precision, recall, fscore, support) = precision_recall_fscore_support(
            y_test, y_pred, average='binary', pos_label=1
        )
    print("Precision  {:3.2%}".format(precision))
    print("Recall     {:3.2%}".format(recall))
    print("F-score    {:3.2%}".format(fscore))
    print("\n")
    print("Classification Report")
    print("---------------------")
    print("{}".format(classification_report(y_test, y_pred)))
    print("Confusion Matrix")
    print("----------------")
    print("{}".format(confusion_matrix(y_test, y_pred)))

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

        doc_class = 0  # label: not_uncertain
        if len(doc['ccue']) > 0:
            doc_class = 1  # label: uncertain

#        print(doc_class)
        x.append(feature_vector)
        y.append(doc_class)
        cnt += 1
        print(cnt)

    x = csr_matrix(np.asarray(x))
    y = np.asarray(y)

#    print(x)
#    print(y)
    return x, y
