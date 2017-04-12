import _pickle
import json
import numpy as np
import os
import sklearn
import sys

from scipy.sparse import csr_matrix

from scripts.convert import *
from scripts.train import *
from features.features import *
from corpora import *

MODEL_FILENAME = os.path.join(os.path.split(__file__)[0], 'uncertainty-svm.p')

CLASSIFIER = _pickle.load(open(MODEL_FILENAME, 'rb'))

def score(document):
    """
    Given a document, calculate it's uncertainty against the pre-trained model.

    Input: [{'text': "I am the walrus."}]
    Output: {'uncertain': confidence, 'not uncertain': confidence}
    """
    global CLASSIFIER
    feature_set = get_features(document)
    feature_vector = [feature_set[f] for f in sorted(feature_set.keys())]
    x = csr_matrix(np.asarray([feature_vector]))
    probs = CLASSIFIER.predict_proba(x)

    probs = {'uncertain': probs[0][1], 'not_uncertain': probs[0][0]}

    return probs

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 1:
        sys.exit(1)
    else:
        if args[0] == "convert":

            WikiConverter(WIKI_OLD, WIKI_RAW, WIKI_NEW).convert()
            FactbankConverter(FACTBANK_OLD, FACTBANK_RAW, FACTBANK_NEW).convert()
            BioBmcConverter(BIO_BMC_OLD, BIO_BMC_RAW, BIO_BMC_NEW).convert()
            BioFlyConverter(BIO_FLY_OLD, BIO_FLY_RAW, BIO_FLY_NEW).convert()
            BioHbcConverter(BIO_HBC_OLD, BIO_HBC_RAW, BIO_HBC_NEW).convert()
        elif args[0] == "train":
            print("Gathering all available training data....")
            all_docs = []
            all_docs += json.loads(open(WIKI_NEW, 'r').read())
            all_docs += json.loads(open(FACTBANK_NEW, 'r').read())
            all_docs += json.loads(open(BIO_BMC_NEW, 'r').read())
            all_docs += json.loads(open(BIO_FLY_NEW, 'r').read())
            all_docs += json.loads(open(BIO_HBC_NEW, 'r').read())
            print(len(all_docs))

            print("Starting to train model...")
            FITTED_SVC = train_svm(all_docs, ntesting=int(args[1]))

            print("Dumping model to disk...")
            _pickle.dump(FITTED_SVC, open("uncertainty-svm.p", "wb"))

            print("Tidying up...")
        elif args[0] == "classify":
            documents = []
            print("Formatting the given documents...")
            with open(args[1], 'r') as doc:
                for line in doc.readlines():
                    documents.append({'text': line, 'ccue': {}})

            uncertain = []
            not_uncertain = []
            for i, document in enumerate(documents):
                probs = score(document)
                uncertain.append(probs['uncertain'])
                not_uncertain.append(probs['not_uncertain'])

                print("\n==== Sentence " + str(i) + ":")
                print("==== " + str(document['text']))
                print("\tP(uncertain) = %.3f" % probs['uncertain'])
                print("\tP(not_uncertain) = %.3f" % probs['not_uncertain'])

            print("\n==== Document:")
            print("\tP(uncertain) = %.3f" % np.mean(uncertain))
            print("\tP(not_uncertain) = %.3f" % np.mean(not_uncertain))

