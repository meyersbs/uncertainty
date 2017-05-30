import collections
import csv
import itertools
import operator
import os
import pickle
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


def aggregate(predictions):
    prediction = 'C'

    frequency = collections.Counter(predictions)
    del frequency['C']

    if frequency:
        predictions = dict()
        for (p, f) in frequency.items():
            if f not in predictions:
                predictions[f] = list()
            predictions[f].append(p)
        predictions = sorted(
                predictions.items(), key=operator.itemgetter(0), reverse=True
            )
        prediction = predictions[0][1][0] \
            if predictions and len(predictions[0][1]) == 1 else 'U'

    return prediction


def dump(obj, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(obj, file)


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


def get_features(data_file):
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


def get_verbs(filepath):
    """
    Return the contents of verbs file pointed to by the filepath argument as a
    dictionary in which the key is the conjugate of a verb and the value is
    uninflected verb form of the conjugate verb.

    For example, {'scolded': 'scold', 'scolding': 'scold'}

    Adapted from code provided in NodeBox:
    https://www.nodebox.net/code/index.php/Linguistics#verb_conjugation
    """
    verbs = dict()
    with open(filepath) as file:
        reader = csv.reader(file)
        for row in reader:
            for verb in row[1:]:
                verbs[verb] = row[0]

    return verbs


def load(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError('No such file: {}'.format(filepath))

    with open(filepath, 'rb') as file:
        return pickle.load(file)


def read_tsv(path):
    lines = list()
    with open(path, 'r') as file:
        for line in file:
            line = line.strip(os.linesep)
            lines.append(line.split('\t') if line else list())
    return lines


def write_tsv(lines, path):
    with open(path, 'w') as file:
        for line in lines:
            file.write('{}{}'.format('\t'.join(line), os.linesep))
