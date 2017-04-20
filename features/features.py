import nltk
import re

from itertools import groupby
from nltk import pos_tag
from nltk.stem.porter import *

STEMMER = PorterStemmer()

GREEK_LOWER = re.compile(u'[αβγδεζηθικλμξπρσςτυφψω]')
GREEK_UPPER = re.compile(u'[ΓΔΘΛΞΠΣΦΨΩ]')
ROMAN_LOWER = re.compile('m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})')
ROMAN_UPPER = re.compile('M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})')

def _format_surface_patterns(pattern_list, indices, offset):
    formatted = {}
    for pattern in pattern_list:
        key = 'pattern_' + str(indices[-1]-offset) + '_' + str(pattern)
        formatted[key] = 1.0
        if indices[-1]-offset == 0:
            key = 'pattern_prefix_' + str(pattern[0])
            formatted[key] = 1.0
        indices.pop()

    return formatted

def clean_word_pattern(word_pattern):
    return ''.join(c for c, _ in groupby(word_pattern))

def get_word_pattern(word):
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

    return clean_word_pattern(pattern)

def get_surface_patterns(sent_list, curr_pos):
    surf_pats = []
    indices = [curr_pos-1, curr_pos, curr_pos+1]

    # Remove indices that are out of bounds.
    indices = list(filter(lambda x: not x < 0, indices))
    indices = list(filter(lambda x: not x >= len(sent_list), indices))

    toks = [sent_list[x] for x in indices]
    for tok in toks:
        surf_pats.append(get_word_pattern(tok))

    return _format_surface_patterns(surf_pats, list(reversed(indices)), curr_pos)

def _format_stems(stem_list, indices, offset):
    formatted = {}
    for stem in stem_list:
        key = 'lemma_' + str(indices[-1]-offset) + '_' + str(stem)
        formatted[key] = 1.0
        indices.pop()

    return formatted

def get_stems(sent_list, curr_pos):
    stems = []
    indices = [curr_pos-2, curr_pos-1, curr_pos, curr_pos+1, curr_pos+2]

    # Remove indices that are out of bounds.
    indices = list(filter(lambda x: not x < 0, indices))
    indices = list(filter(lambda x: not x >= len(sent_list), indices))

    stems = [STEMMER.stem(sent_list[x]) for x in indices]
    return _format_stems(stems, list(reversed(indices)), curr_pos)

def _format_prefixes(prefix_list):
    formatted = {}
    indices = [5, 4, 3]
    for prefix in prefix_list:
        key = 'prefix_' + str(indices[-1]) + '_' + str(prefix)
        formatted[key] = 1.0
        indices.pop()

    return formatted

def get_prefixes(token):
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

    return _format_prefixes(prefixes)

def _format_suffixes(suffix_list):
    formatted = {}
    indices = [5, 4, 3]
    for suffix in suffix_list:
        key = 'suffix_' + str(indices[-1]) + '_' + str(suffix)
        formatted[key] = 1.0
        indices.pop()

    return formatted

def get_suffixes(token):
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

    return _format_suffixes(suffixes)

def _format_pos_tags(tags_list, indices, offset):
    formatted = {}
    for tag in tags_list:
        key = 'pos_' + str(indices[-1]-offset) + '_' + str(tag)
        formatted[key] = 1.0
        indices.pop()

    return formatted

def get_pos_tags(sent_list, curr_pos):
    tags = []
    indices = [curr_pos-2, curr_pos-1, curr_pos, curr_pos+1, curr_pos+2]

    # Remove indices that are out of bounds.
    indices = list(filter(lambda x: not x < 0, indices))
    indices = list(filter(lambda x: not x >= len(sent_list), indices))

    tags = [pos_tag([sent_list[x]])[0][1] for x in indices]
    return _format_pos_tags(tags, list(reversed(indices)), curr_pos)

def get_chunks(sent_list, curr_pos, treeparse=None):
    if treeparse is None:
        treeparse = SPLAT(" ".join(sent_list)).treestrings()

    return treeparse
    # TODO: Figure out how to use C&C Tools or an NLTK Chunker.

def get_combinations(sent_list, curr_pos):
    # TODO: Implement Feature 8 from the README.
    pass

def get_features(sentence, treeparse=None):
    feature_dict = {}
    sent_list = sentence.split()
    for i, token in enumerate(sent_list):
        feature_dict[token] = {}
        feature_dict[token].update(get_stems(sent_list, i))
        feature_dict[token].update(get_prefixes(token))
        feature_dict[token].update(get_suffixes(token))
        feature_dict[token].update(get_pos_tags(sent_list, i))
        feature_dict[token].update(get_surface_patterns(sent_list, i))
        print(get_chunks(sent_list, i, treeparse))

    return feature_dict

sentence = "Cells in Regulating Cellular Immunity"
print(get_features(sentence))
