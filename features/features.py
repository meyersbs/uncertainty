import nltk
import pprint
import re
import warnings

from SuperChunker import *

from itertools import groupby
from nltk import pos_tag
from nltk.stem.porter import *

warnings.filterwarnings("ignore", category=FutureWarning)

CHUNKER = SuperChunker()
STEMMER = PorterStemmer()
PRINTER = pprint.PrettyPrinter(indent=4, width=80)

GREEK_LOWER = re.compile(u'[αβγδεζηθικλμξπρσςτυφψω]')
GREEK_UPPER = re.compile(u'[ΓΔΘΛΞΠΣΦΨΩ]')
#ΓΔΘΛΞΠΣΦΨΩ αβγδεζηθικλμξπρςστυφψω
ROMAN_LOWER = re.compile(
                    'm{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})'
                )
ROMAN_UPPER = re.compile(
                    'M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})'
                )

#### FEATURE 1 - PREFIXES OF LENGTH 3-5 ########################################
#### Example:
####    Input: 	'Distinct'
####    Output: ['Dis', 'Dist', 'Disti']
####
#### Formatted Output:
####    ['prefix_3_Dis', 'prefix_4_Dist', 'prefix_5_Disti']
################################################################################
def _format_feature1(prefix_list):
    """
    Given a list of prefix strings, return a dictionary. The keys are formatted
    as 'prefix_', concatenated with the length of the current prefix string,
    concatenated with the current prefix string. The values for every key will
    be 1.0.

    Assertions:
        len(prefix_list) == 3
    """
    formatted = {}
    indices = [5, 4, 3]
    for prefix in prefix_list:
        key = 'prefix_' + str(indices[-1]) + '_' + str(prefix)
        formatted[key] = 1.0
        indices.pop()

    return formatted

def get_feature1(token):
    prefixes = []
    # Example: token = 'of', token = 'the'
    if len(token) <= 3:
        prefixes.append(token) # 'of', 'the'
        prefixes.append(token) # 'of', 'the'
        prefixes.append(token) # 'of', 'the'
    # Example: token = 'that'
    elif len(token) == 4:
        prefixes.append(token[0:3]) # 'tha'
        prefixes.append(token[0:4]) # 'that'
        prefixes.append(token[0:4]) # 'that'
    # Example: token = 'Bagels'
    else:
        i = 3
        while(i < 6 and i < len(token)):
            prefixes.append(token[0:i])
            i += 1;

    return _format_feature1(prefixes)

#### FEATURE 2- SUFFIXES OF LENGTH 3-5 #########################################
#### Example:
####    Input:	'Distinct'
####    Output:	['nct', 'inct', 'tinct']
####
#### Formatted Output:
####    ['suffix_3_nct', 'suffix_4_inct', 'suffix_5_tinct']
################################################################################
def _format_feature2(suffix_list):
    """
    Given a list of suffix strings, return a dictionary. The keys are formatted
    as 'suffix_', concatenated with the length of the current suffix string,
    concatenated with the current suffix string. The values for every key will
    be 1.0.

    Assertions:
        len(suffix_list) == 3
    """
    formatted = {}
    indices = [5, 4, 3]
    for suffix in suffix_list:
        key = 'suffix_' + str(indices[-1]) + '_' + str(suffix)
        formatted[key] = 1.0
        indices.pop()

    return formatted

def get_feature2(token):
    suffixes = []
    # Example: token = 'of', token = 'the'
    if len(token) <= 3:
        suffixes.append(token) # 'of', 'the'
        suffixes.append(token) # 'of', 'the'
        suffixes.append(token) # 'of', 'the'
    # Example: token = 'that'
    elif len(token) == 4:
        suffixes.append(token[-3:]) # 'hat'
        suffixes.append(token) # 'that'
        suffixes.append(token) # 'that'
    # Example: token = 'Bagels'
    else:
        i = 3
        while(i < 6 and i < len(token)):
            suffixes.append(token[-i:])
            i += 1;

    return _format_feature2(suffixes)

#### FEATURE 3 - STEMS/LEMMAS W/ WINDOW OF LENGTH 2 ############################
#### Example:
####    Input:
####        sent_list = ["Cells", "in", "Regulating", "Cellular", "Immunity"]
####        curr_pos = 2
####    Output:
####        ["cell", "in", "regulate", "cellular", "immun"]
#### Formatted Output:
####    ["lemma_-2_cell", "lemma_-1_in", "lemma_0_regulate, "lemma_1_cellular",
####     "lemma_2_immun"]
################################################################################
def _format_feature3(stem_list, indices, offset):
    formatted = {}
    for stem in stem_list:
        key = 'lemma_' + str(indices[-1]-offset) + '_' + str(stem).lower()
        formatted[key] = 1.0
        indices.pop()

    return formatted

def get_feature3(sent_list, curr_pos):
    stems = []
    indices = [curr_pos-2, curr_pos-1, curr_pos, curr_pos+1, curr_pos+2]

    # Remove indices that are out of bounds.
    indices = list(filter(lambda x: not x < 0, indices))
    indices = list(filter(lambda x: not x >= len(sent_list), indices))

    stems = [STEMMER.stem(sent_list[x]) for x in indices]
    return _format_feature3(stems, list(reversed(indices)), curr_pos)

#### FEATURES 4 & 5 - SURFACE PATTERNS W/ WINDOW OF LENGTH 1 ###################
#### Example:
####    Input:
####        sent_list = ["Cells", "in", "Regulating", "Cellular", "Immunity"]
####        curr_pos = 2
####    Output: ['a', 'Aa', 'Aa']
####
#### Formatted Output:
####    ['pattern_-1_a', 'pattern_0_Aa', 'pattern_1_Aa', 'pattern_prefix_A']
################################################################################
def _format_features4_5(pattern_list, indices, offset):
    formatted = {}
    for pattern in pattern_list:
        key = 'pattern_' + str(indices[-1]-offset) + '_' + str(pattern)
        formatted[key] = 1.0
        if indices[-1]-offset == 0:
            key = 'pattern_prefix_' + str(pattern[0])
            formatted[key] = 1.0
        indices.pop()

    return formatted

def _clean_word_pattern(word_pattern):
    return ''.join(c for c, _ in groupby(word_pattern))

def _get_word_pattern(word):
    pattern = ""
    chars = [c for c in word]
    # TODO: How do we distinguish between a sequence of Roman Numerals and a
    # sequence of letters?
    for i, c in enumerate(chars):
        if re.search(r'[A-Z]', c):
            pattern += 'A'
        elif re.search(r'[a-z]', c):
            pattern += 'a'
        elif re.search(r'[0-9]', c):
            pattern += '0'
        elif GREEK_UPPER.search(c):
            pattern += 'G'
        elif GREEK_LOWER.search(c):
            pattern += 'g'
        elif ROMAN_UPPER.search(c):
            pattern += 'R'
        elif ROMAN_LOWER.search(c):
            pattern += 'r'
        elif not c.isalnum():
            pattern += '!'

    return _clean_word_pattern(pattern)

def get_features4_5(sent_list, curr_pos):
    surf_pats = []
    indices = [curr_pos-1, curr_pos, curr_pos+1]

    # Remove indices that are out of bounds.
    indices = list(filter(lambda x: not x < 0, indices))
    indices = list(filter(lambda x: not x >= len(sent_list), indices))

    toks = [sent_list[x] for x in indices]
    for tok in toks:
        surf_pats.append(_get_word_pattern(tok))

    return _format_features4_5(surf_pats, list(reversed(indices)), curr_pos)

#### FEATURE 6 - PART-OF-SPEECH TAGS W/ WINDOW OF LENGTH 2 #####################
#### Example:
####    Input:
####        sent_list = ["Cells", "in", "Regulating", "Cellular", "Immunity"]
####        curr_pos = 2
####    Output:	["NNS", "IN", "VBG", "JJ", "NN"]
####
#### Formatted Output:
####    ["pos_-2_NNS", "pos_-1_IN", "pos_0_VBG", "pos_1_JJ", "pos_2_NN"]
################################################################################
def _format_feature6(tags_list, indices, offset):
    formatted = {}
    for tag in tags_list:
        key = 'pos_' + str(indices[-1]-offset) + '_' + str(tag).upper()
        formatted[key] = 1.0
        indices.pop()

    return formatted

def get_feature6(sent_list, curr_pos):
    tags = []
    indices = [curr_pos-2, curr_pos-1, curr_pos, curr_pos+1, curr_pos+2]

    # Remove indices that are out of bounds.
    indices = list(filter(lambda x: not x < 0, indices))
    indices = list(filter(lambda x: not x >= len(sent_list), indices))

    tags = [pos_tag([sent_list[x]])[0][1] for x in indices]
    return _format_feature6(tags, list(reversed(indices)), curr_pos)

#### FEATURE 7 - CHUNKS W/ WINDOW OF LENGTH 2 ##################################
#### Example:
####    Input:
####        sent_list = ["Cells", "in", "Regulating", "Cellular", "Immunity"]
####        feats = [
####            {'stem':'Cell','pos':'NNS','chunk':'B-NP','token':'Cells'},
####            {'stem':'in','pos':'IN','chunk':'O','token':'in'},
####            {'stem':'Regul','pos':'VBG','chunk':'O','token':'Regulating'},
####            {'stem':'Cellular','pos':'JJ','chunk':'B-NP','token':'Cellular'},
####            {'stem':'Immun','pos':'NN','chunk':'I-NP','token':'Immunity'}
####        ]
####        curr_pos = 2
####    Output:	["I-np", "B-pp", "B-vp", "B-np", "I-np"]
####
#### Formatted Output:
####    ["chunk_-2_I-np", "chunk_-1_B-pp", "chunk_0_B-vp", "chunk_1_B-np",
####     "chunk_2_I-np"]
################################################################################
def _format_feature7(chunk_list, indices, offset):
    formatted = {}
    for chunk in chunk_list:
        key = 'chunk_' + str(indices[-1]-offset) + '_' + str(chunk)
        formatted[key] = 1.0
        indices.pop()

    return formatted

def get_feature7(sent_list, feats, curr_pos):
    chunks = []
    indices = [curr_pos-2, curr_pos-1, curr_pos, curr_pos+1, curr_pos+2]

    # Remove indices that are out of bounds.
    indices = list(filter(lambda x: not x < 0, indices))
    indices = list(filter(lambda x: not x >= len(sent_list), indices))

    chunks = [feats[x]["chunk"] for x in indices]
    return _format_feature7(chunks, list(reversed(indices)), curr_pos)

#### FEATURE 8A - STEM-CHUNK W/ WINDOW OF LENGTH 1  ############################
#### Example:
####    Input:
####        sent_list = ["Cells", "in", "Regulating", "Cellular", "Immunity"]
####        feats = [
####            {'stem':'Cell','pos':'NNS','chunk':'B-NP','token':'Cells'},
####            {'stem':'in','pos':'IN','chunk':'O','token':'in'},
####            {'stem':'Regul','pos':'VBG','chunk':'O','token':'Regulating'},
####            {'stem':'Cellular','pos':'JJ','chunk':'B-NP','token':'Cellular'},
####            {'stem':'Immun','pos':'NN','chunk':'I-NP','token':'Immunity'}
####        ]
####        curr_pos = 2
####    Formatted Output:
####        ["L_-1_in_C_0_O", "L_0_regulate_C_0_O", "L_1_cellular_C_0_O"]
################################################################################
def _format_feature8A(stem_list, indices, offset, chunk):
    formatted = {}
    for stem in stem_list:
        key = 'L_' + str(indices[-1]-offset) + "_" + stem.lower() + "_C_0_" + chunk
        formatted[key] = 1.0
        indices.pop()

    return formatted

def get_feature8A(sent_list, feats, curr_pos):
    curr_chunks = []
    indices = [curr_pos-1, curr_pos, curr_pos+1]

    # Remove indices that are out of bounds.
    indices = list(filter(lambda x: not x < 0, indices))
    indices = list(filter(lambda x: not x >= len(sent_list), indices))

    curr_chunks = [feats[x]["stem"] for x in indices]
    return _format_feature8A(curr_chunks, list(reversed(indices)), \
                             curr_pos, feats[curr_pos]["chunk"])

#### FEATURE 8B - STEM-POS W/ WINDOW OF LENGTH 1  ##############################
#### Example:
####    Input:
####        sent_list = ["Cells", "in", "Regulating", "Cellular", "Immunity"]
####        feats = [
####            {'stem':'Cell','pos':'NNS','chunk':'B-NP','token':'Cells'},
####            {'stem':'in','pos':'IN','chunk':'O','token':'in'},
####            {'stem':'Regul','pos':'VBG','chunk':'O','token':'Regulating'},
####            {'stem':'Cellular','pos':'JJ','chunk':'B-NP','token':'Cellular'},
####            {'stem':'Immun','pos':'NN','chunk':'I-NP','token':'Immunity'}
####        ]
####        curr_pos = 2
####    Formatted Output:
####        ["L_-1_in_P_0_VBG", "L_0_regulate_P_0_VBG", "L_1_cellular_P_0_VBG"]
################################################################################
def _format_feature8B(stem_list, indices, offset, pos):
    formatted = {}
    for stem in stem_list:
        key = 'L_' + str(indices[-1]-offset) + "_" + stem.lower() + "_P_0_" + pos
        formatted[key] = 1.0
        indices.pop()

    return formatted

def get_feature8B(sent_list, feats, curr_pos):
    curr_chunks = []
    indices = [curr_pos-1, curr_pos, curr_pos+1]

    # Remove indices that are out of bounds.
    indices = list(filter(lambda x: not x < 0, indices))
    indices = list(filter(lambda x: not x >= len(sent_list), indices))

    curr_chunks = [feats[x]["stem"] for x in indices]
    return _format_feature8B(curr_chunks, list(reversed(indices)), \
                             curr_pos, feats[curr_pos]["pos"])

################################################################################
def get_token_features(sent_list):
    chunks = CHUNKER.parse(pos_tag(sent_list))
    for i, chunk in enumerate(chunks):
        stem = STEMMER.stem(chunk["token"])
        chunk.update({"stem": stem})

    return chunks

def get_features(sentence):
    feature_dict = {}
    sent_list = sentence.split()
    feats = get_token_features(sent_list)
    prepend="000"
    for i, token in enumerate(sent_list):
        if i > 999:
            prepend=""
        elif i > 99:
            prepend="0"
        elif i > 9:
            prepend="00"

        feature_dict[prepend + str(i) + "_" + token] = {}
        feature_dict[prepend + str(i) + "_" + token].update(get_feature1(token))
        feature_dict[prepend + str(i) + "_" + token].update(get_feature2(token))
        feature_dict[prepend + str(i) + "_" + token].update(get_feature3(sent_list, i))
        feature_dict[prepend + str(i) + "_" + token].update(get_features4_5(sent_list, i))
        feature_dict[prepend + str(i) + "_" + token].update(get_feature6(sent_list, i))
        feature_dict[prepend + str(i) + "_" + token].update(get_feature7(sent_list, feats, i))
        feature_dict[prepend + str(i) + "_" + token].update(get_feature8A(sent_list, feats, i))
        feature_dict[prepend + str(i) + "_" + token].update(get_feature8B(sent_list, feats, i))

    return feature_dict
