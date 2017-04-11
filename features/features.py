import nltk
import re
import sys

from collections import defaultdict as ddict

from splat.gramminators.FullNGramminator import *
from splat.taggers.NLTKPOSTagger import *
from splat.Util import get_pos_counts

from nltk.stem.porter import PorterStemmer

from features import UNI_HEDGES, UNI_WEASELS, UNI_PEACOCKS, \
                     BI_HEDGES, BI_WEASELS, BI_PEACOCKS, \
                     TRI_HEDGES, TRI_WEASELS, TRI_PEACOCKS

REGEX = re.compile("[^\w\s&\-']")

def get_pos_freqs(document):
    """
    Given a sentence as a string, use NLTK to split and tag parts of speech.
    Then combine duplicate entries in order to get frequencies.

    For Example:
        Input: {'text': "I am the walrus. I am the eggman.", 'id': 4}
        Output: {'VBP': 2, '.': 2, 'NN': 2, 'DT': 2, 'PRP': 2}
    """
    return get_pos_counts(NLTKPOSTagger().tag(document['text']))

def get_ngrams(document):
    """
    Given a dictionary of the form, {'text': "sample text", ...}, return three
    lists of ngrams (uni-, bi-, and tri-), where each item in a list is a single
    string.

    For Example:
        Input: {'text': "I am the walrus.", 'id': 4}
        Output:
            unigrams: ['i', 'am', 'the', 'walrus']
            bigrams: ['i am', 'am the', 'the walrus']
            trigrams: ['i am the', 'am the walrus']
    """
    ng = FullNGramminator()
    unigrams = [" ".join(n) for n in ng.unigrams(preprocess(document['text']))]
    bigrams = [" ".join(n) for n in ng.bigrams(preprocess(document['text']))]
    trigrams = [" ".join(n) for n in ng.trigrams(preprocess(document['text']))]

    return unigrams, bigrams, trigrams

def flatten_dict(d):
    """
    Given a dictionary of the following form, return a dictionary with the same
    keys, but values equal to the length of the corresponding value in the
    original dictionary.

    For Example:
        Input: {'i': [True, True], 'am': [True, True], 'the': [True, True],
                'walrus': [True], 'eggman': [True]}
        Output: {'i': 2, 'am': 2, 'the': 2, 'walrus': 1, 'eggman': 1}
    """
    new_d = {}
    for k, v in d.items():
        new_d[k] = len(v)

    return new_d

def count_hedges(ngrams):
    """ Return the number of hedges present in each given list. """
    global UNI_HEDGES, BI_HEDGES, TRI_HEDGES
    uni_hedges, bi_hedges, tri_hedges = (ddict(list),)*3
    for u in ngrams[0]:
        if u in UNI_HEDGES:
            uni_hedges[u].append(True)
    for b in ngrams[1]:
        if b in BI_HEDGES:
            bi_hedges[b].append(True)
    for t in ngrams[2]:
        if t in TRI_HEDGES:
            tri_hedges[t].append(True)

    uni_hedges = flatten_dict(uni_hedges)
    bi_hedges = flatten_dict(bi_hedges)
    tri_hedges = flatten_dict(tri_hedges)

    return uni_hedges, bi_hedges, tri_hedges

def count_weasels(ngrams):
    """ Return the number of weasels present in each given list. """
    global UNI_WEASELS, BI_WEASELS, TRI_WEASELS
    uni_weasels, bi_weasels, tri_weasels = (ddict(list),)*3
    for u in ngrams[0]:
        if u in UNI_WEASELS:
            uni_weasels[u].append(True)
    for b in ngrams[1]:
        if b in BI_WEASELS:
            bi_weasels[b].append(True)
    for t in ngrams[2]:
        if t in TRI_WEASELS:
            tri_weasels[t].append(True)

    uni_weasels = flatten_dict(uni_weasels)
    bi_weasels = flatten_dict(bi_weasels)
    tri_weasels = flatten_dict(tri_weasels)

    return uni_weasels, bi_weasels, tri_weasels

def count_peacocks(ngrams):
    """ Return the number of peacocks present in each given list. """
    global UNI_PEACOCKS, BI_PEACOCKS, TRI_PEACOCKS
    uni_peacocks, bi_peacocks, tri_peacocks = (ddict(list),)*3
    for u in ngrams[0]:
        if u in UNI_PEACOCKS:
            uni_peacocks[u].append(True)
    for b in ngrams[1]:
        if b in BI_PEACOCKS:
            bi_peacocks[b].append(True)
    for t in ngrams[2]:
        if t in TRI_PEACOCKS:
            tri_peacocks[t].append(True)

    uni_peacocks = flatten_dict(uni_peacocks)
    bi_peacocks = flatten_dict(bi_peacocks)
    tri_peacocks = flatten_dict(tri_peacocks)

    return uni_peacocks, bi_peacocks, tri_peacocks

def get_lexical_features(document):
    """
    Given a document of the form, {'text': "I am the walrus.", 'id': 4},
    return a dictionary of features related to unigrams, bigrams, trigrams,
    hedges, weasels, peacocks, and parts-of-speech.
    """
    features = {'unigrams': {}, 'bigrams': {}, 'trigrams': {}}

    # Get unigrams, bigrams, and trigrams
    ngrams = get_ngrams(document)
    features['unigrams']['raw'] = ngrams[0]
    features['bigrams']['raw'] = ngrams[1]
    features['trigrams']['raw'] = ngrams[2]

    # Get hedges and their frequencies
    hedges = count_hedges(ngrams)
    features['unigrams']['hedges'] = hedges[0]
    features['bigrams']['hedges'] = hedges[1]
    features['trigrams']['hedges'] = hedges[2]

    # Get weasels and their frequencies
    weasels = count_weasels(ngrams)
    features['unigrams']['weasels'] = weasels[0]
    features['bigrams']['weasels'] = weasels[1]
    features['trigrams']['weasels'] = weasels[2]

    # Get peacocks and their frequencies
    peacocks = count_peacocks(ngrams)
    features['unigrams']['peacocks'] = peacocks[0]
    features['bigrams']['peacocks'] = peacocks[1]
    features['trigrams']['peacocks'] = peacocks[2]

    features['pos'] = get_pos_freqs(document)

    return features

def get_semantic_features(document, lexical_features):
    """
    Given a dictionary of lexical features obtained by get_lexical_features(),
    translate those features to semantic features.
    """
    features = {
            'is_speculation': 0, 'is_negation': 0, 'is_modal': 0, 'is_hypo': 0,
            'is_epistemic': 0, 'is_doxastic': 0, 'is_investigation': 0,
            'is_condition': 0, 'has_hedges': 0, 'has_weasels': 0,
            'has_peacocks': 0
        }
    '''
    for i in range(len(UNI_HEDGES)):
        features['has_hedge_' + UNI_HEDGES[i].replace(" ", "_")] = 0

    for i in range(len(BI_HEDGES)):
        features['has_hedge_' + BI_HEDGES[i].replace(" ", "_")] = 0

    for i in range(len(TRI_HEDGES)):
        features['has_hedge_' + TRI_HEDGES[i].replace(" ", "_")] = 0
    '''
    stemmer = PorterStemmer()

    for entry in ['unigram', 'bigram', 'trigram']:
        for k, v in lexical_features[str(entry) + 's'].items():
            if k == 'hedges' and len(v) > 0:
                features['has_hedges'] = 1
#                for item in v:
#                    features['has_hedge_' + item.replace(" ", "_")] = 1
            if k == 'weasels' and len(v) > 0:
                features['has_weasels'] = 1
#                for item in v:
#                    features['has_weasel_' + item.replace(" ", "_")] = 1
            if k == 'peacocks' and len(v) > 0:

                features['has_peacocks'] = 1
#                for item in v:
#                    features['has_peacock_' + item.replace(" ", "_")] = 1
#            if k == 'raw' and len(v) > 0:
#                for item in v:
#                    features['has_' + entry + '_' + item.replace(" ", "_")] = 1
#                    if entry == 'unigram':
#                        features['has_stem_' + stemmer.stem(item)] = 1

#    for key in lexical_features['pos'].keys():
#        features['has_pos_' + key] = 1

    for k, v in document['ccue'].items():
        if k is not None:
            #print(k)
            if re.search(r'speculation', k) is not None:
                features['is_speculation'] += 1
            if re.search(r'negation', k) is not None:
                features['is_negation'] += 1
            if re.search(r'modal', k) is not None:
                features['is_modal'] += 1
            if re.search(r'hypo', k) is not None:
                features['is_hypo'] += 1
            if re.search(r'probable', k) is not None:
                features['is_epistemic'] += 1
            if re.search(r'doxastic', k) is not None:
                features['is_doxastic'] += 1
            if re.search(r'investigation', k) is not None:
                features['is_investigation'] += 1
            if re.search(r'condition', k) is not None:
                features['is_condition'] += 1

    return features

def get_features(document):
    lexical_features = get_lexical_features(document)
    semantic_features = get_semantic_features(document, lexical_features)

    return semantic_features

def preprocess(sent):
    global REGEX
    clean_sent = REGEX.sub('', sent.lower())
    clean_sent = clean_sent.replace('&', 'and')
    clean_sent = clean_sent.replace('_', '-')

    return clean_sent

def features(documents):
    feature_dict = {}

    for sent in documents:
        temp_dict = {}
        temp_dict['sent'] = sent['text']
        temp_dict.update(get_features(sent))
        feature_dict[len(feature_dict)] = temp_dict
#        print("=====")
#        print(temp_dict)

    return feature_dict
