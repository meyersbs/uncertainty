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

    return surf_pats

def get_stems(sent_list, curr_pos):
    stems = []
    indices = [curr_pos-2, curr_pos-1, curr_pos, curr_pos+1, curr_pos+2]

    # Remove indices that are out of bounds.
    indices = list(filter(lambda x: not x < 0, indices))
    indices = list(filter(lambda x: not x >= len(sent_list), indices))

    stems = [STEMMER.stem(sent_list[x]) for x in indices]
    return stems

def get_prefixes(token):
    prefixes = []
    i = 3
    while(i < 6 and i < len(token)):
        prefixes.append(token[0:i])
        i += 1;

    return prefixes

def get_suffixes(token):
    suffixes = []
    i = 3
    while(i < 5 and i < len(token)):
        suffixes.append(token[-i:])
        i += 1;

    return suffixes

def get_pos_tags(sent_list, curr_pos):
    tags = []
    indices = [curr_pos-2, curr_pos-1, curr_pos, curr_pos+1, curr_pos+2]

    # Remove indices that are out of bounds.
    indices = list(filter(lambda x: not x < 0, indices))
    indices = list(filter(lambda x: not x >= len(sent_list), indices))

    tags = [pos_tag(sent_list[x])[1] for x in indices]
    return tags

def get_chunks(sent_list, curr_pos):
    # TODO: Figure out how to use C&C Tools or an NLTK Chunker.
    pass

def get_combinations(sent_list, curr_pos):
    # TODO: Implement Feature 7 from the README.
