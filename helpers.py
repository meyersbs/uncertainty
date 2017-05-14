import collections
import itertools
import re

GREEK_LOWER = re.compile(u'[αβγδεζηθικλμξπρσςτυφψω]')
GREEK_UPPER = re.compile(u'[ΓΔΘΛΞΠΣΦΨΩ]')
ENGLISH_LOWER = re.compile(r'[a-z]')
ENGLISH_UPPER = re.compile(r'[A-Z]')
DIGIT = re.compile(r'[0-9]')
ROMAN_LOWER = re.compile(
        'm{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})'
    )
ROMAN_UPPER = re.compile(
        'M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})'
    )


def get_context(index, elements, size=2):
    if index < 0 or index >= len(elements):
        raise Exception('index must be positive and within elements\' bounds.')

    context = [None] * (size * 2)

    offsets = list(range(-size, size + 1))
    offsets.remove(0)

    indices = [index + offset for offset in offsets]
    for (i, j) in enumerate(indices):
        if j >= 0 and j < len(elements):
            context[i] = elements[j]

    return collections.OrderedDict(zip(offsets, context))


def get_wordpattern(word):
    pattern = ''.join(get_charpattern(character) for character in word)
    return ''.join(c for c, _ in itertools.groupby(pattern))


def get_charpattern(character):
    # TODO: How do we distinguish between a sequence of Roman Numerals and
    #       a sequence of letters?
    if not character.isalnum():
        return '!'
    elif ENGLISH_UPPER.search(character):
        return 'A'
    elif ENGLISH_LOWER.search(character):
        return 'a'
    elif DIGIT.search(character):
        return '0'
    elif GREEK_UPPER.search(character):
        return 'G'
    elif GREEK_LOWER.search(character):
        return 'g'
    elif ROMAN_UPPER.search(character):
        return 'R'
    elif ROMAN_LOWER.search(character):
        return 'r'
