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
    # TODO: Cleanup this nonsense.
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
    # The features listed below should not be used, as they indirectly define
    #    whether or not a sentence is uncertain. Using them results in 100%
    #    precision, recall, and f1-score.
    #
    #    1) 'is_speculation' 	2) 'is_negation'	3) 'is_modal'
    #    4) 'is_hypo'		5) 'is_epistemic'	6) 'is_doxastic'
    #    7) 'is_investigation'	8) 'is_condition'
    features = {
            'has_hedges': 0, 'has_weasels': 0, 'has_peacocks': 0,
            'word_count': len(document['text'].split())#,
            #'unigrams': len(lexical_features['unigrams']['raw']),
            #'bigrams': len(lexical_features['bigrams']['raw']),
            #'trigrams': len(lexical_features['trigrams']['raw'])
        }
    stemmer = PorterStemmer()

    for entry in ['unigram', 'bigram', 'trigram']:
        for k, v in lexical_features[str(entry) + 's'].items():
            if k == 'hedges' and len(v) > 0:
                features['has_hedges'] = 1
                # Individual hedges could be used as features, but we need to
                #    include all of the available hedges as keys in the
                #    feature dictionary for every sentence.
                #for item in v:
                #    features['has_hedge_' + item.replace(" ", "_")] = 1
            if k == 'weasels' and len(v) > 0:
                features['has_weasels'] = 1
                # Individual weasels could be used as features, but we need to
                #    include all of the available weasels as keys in the
                #    feature dictionary for every sentence.
                #for item in v:
                #    features['has_weasel_' + item.replace(" ", "_")] = 1
            if k == 'peacocks' and len(v) > 0:
                features['has_peacocks'] = 1
                # Individual peacocks could be used as features, but we need to
                #    include all of the available peacocks as keys in the
                #    feature dictionary for every sentence.
                #for item in v:
                #    features['has_peacock_' + item.replace(" ", "_")] = 1

            # Potentially, we could include every unique unigram, bigram, and
            #    trigram as features, but then we would need to write an
            #    external function to ensure that all feature dicts contain an
            #    entry for every unique ngram in the document set -- this would
            #    be incredibly verbose and likely of little value.
            #if k == 'raw' and len(v) > 0:
            #    for item in v:
            #        features['has_' + entry + '_' + item.replace(" ", "_")] = 1
                    # Uncomment to unclude unique stems as features. This would
                    #    also require an external function to propogate all of
                    #    the stems in the document set.
                    #if entry == 'unigram':
                    #    features['has_stem_' + stemmer.stem(item)] = 1

    # Additionally, we could include all of the parts of speech throughout the
    #    document set as features, but we would either need to write an
    #    external propogation function, or include every possible POS tag as a
    #    feature.
    #for key in lexical_features['pos'].keys():
    #    features['has_pos_' + key] = 1

    # Uncomment this to include the semantic uncertainty labels as features. As
    #    previously described, this results in a classifier with 100% precision,
    #    recall, and f1-score.
    '''
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
    '''
    return features

def get_features(document):
    """
    Return a dictionary of features that can be converted to vectors for
    classification purposes.
    """
    lexical_features = get_lexical_features(document)
    semantic_features = get_semantic_features(document, lexical_features)

    return semantic_features

def preprocess(sent):
    """
    Clean up each sentence by 1) removing punctuation, 2) replacing ampersands
    with 'and', and 3) replacing underscores with hyphens.
    """
    global REGEX
    clean_sent = REGEX.sub('', sent.lower())
    clean_sent = clean_sent.replace('&', 'and')
    clean_sent = clean_sent.replace('_', '-')

    return clean_sent
