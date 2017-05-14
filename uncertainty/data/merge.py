import csv
import json
import traceback
import re
import sys
import warnings

from ..sentence import *
from ..word import *

warnings.filterwarnings("ignore", category=FutureWarning, module="__main__")

################################################################################
#### HELPER FUNCTIONS ##########################################################
################################################################################

def _clean_sent(sent):
    """
    This preprocessing function takes in a sentence object, grabs just the
    actual sentence, and runs three regex substitutions to clean up the
    sentence for querying.
    """
    s = re.sub(r'\s([\.,!?\)\]])', r'\1', sent.get_sentence().lower())
    s = re.sub(r'([\(\[])\s', r'\1', s)
    s = re.sub(r'(\s’[\ss])', r'’', s, re.UNICODE)
    return s

def _get_sent_label(labels, sent):
    """
    Given a list of uncertainty cues and a sentence object, return a label
    representing the classification of the entire sentence and each token within
    that sentence.

    Favor is given to the five uncertainty classes; if any token within the
    sentence is labeled as 'U', 'E', 'I', 'D', or 'N', the entire sentence is
    considered to be whichever label that occurs most frequently within the
    sentence. This decision is based on that made by Vincze et al. in their
    binary classifier. Only if there are no occurences of the five uncertainty
    labels within a sentence is the sentence classified as 'C'.
    """
    label_map = {"possible": "E", "probable": "E", "epistemic": "E",
                 "doxastic": "D", "investigation": "I", "condition": "N",
                 "certain": "C", "uncertain": "U"}
    word_labels = []
    for i, word in enumerate(sent.get_words()):
        word_labels.append(word.binary_label)
        if type(labels) != str:
            for k, v in labels.items():
                if word.word in v:
                    word_labels[i] = label_map[k.strip().rstrip("_").split("_")[-1].strip()]
                    break

    print("RAW:\t" + str(labels))
    if not bool(labels):
        return sent.get_label(), word_labels
    else:
        labs = {"E": 0.0, "D": 0.0, "I": 0.0, "N": 0.0, "C": 0.0, "U": 0.0}

        for k, v in labels.items():
            label = label_map[k.strip().rstrip("_").split("_")[-1].strip()]
            labs[label] += len(v)

        max_val = max(labs.values())
        max_keys = []
        for k, v in labs.items():
            if v == max_val:
                max_keys.append(k)

        if len(max_keys) == 1:
            return max_keys[0], word_labels
        else:
            return "U", word_labels

def _get_lines(filepath):
    """ Given a filepath, return a list of lines within that file. """
    lines = None
    with open(filepath) as file:
        lines = file.readlines()
    return lines


def _get_sentences(filepath):
    """ Given a filepath, return a list of sentences within that file. """
    sentences = list()
    _lines = list()
    for line in _get_lines(filepath):
        if line.strip() == '':  # End of Sentence
            sentences.append(_lines)
            _lines = list()
            continue
        _lines.append(line)
    return sentences

################################################################################
#### MAIN FUNCTION #############################################################
################################################################################

def merge_data(json_data, tagged_data):
    """
    There are two datasets associated with the work of Vincze et al [0].

    The first [1A] consists of a collection of XML objects that contain the text
    of a sentence and a list of uncertainty cues (with their categorization)
    that are within the sentence, if any. It has been parsed into a more
    intuitive JSON [1B] structure (at the expense of harddrive space).

    The second [2] consists of tab-delineated lists of tokens from [1A]. Each
    line in the files contains at least five columns: 1) an ID, 2) the raw
    token, 3) the lemma of the token, 4) the part-of-speech tag for the token,
    and 5) a label of certain or uncertain for the token. The tokens are in
    order by sentence, with each sentence separated by an empty line. All
    columns following the first five contain preparsed features used to train
    the binary classifier described in Vincze et al [0].

    This function parses both datasets (which contain the same sentences), and
    matches the uncertainty cues from [1B] to the tokens in [2]. This is not a
    trivial task; since [2] does not actually contain the raw sentences, the
    tokens had to be parsed into sentences and compared against those in [1B].

    This function results in a new file /data/merged_data [3] that is formatted
    in the same manner as [2], but contains a sixth column denoting a specific
    type of uncertainty (epistemic, doxastic, investigation, condition, other)
    when applicable. This new file [3] was used to train the multiclass models
    contained in this codebase.

    [0 ] http://doktori.bibl.u-szeged.hu/2291/1/Vincze_Veronika_tezis.pdf
    [1A] http://rgai.inf.u-szeged.hu/project/nlp/uncertainty/uncertainty.zip
    [1B] http://people.rc.rit.edu/~bsm9339/corpora/szeged_uncertainty/szeged_uncertainty_json.tar.gz
    [2 ] http://rgai.inf.u-szeged.hu/project/nlp/uncertainty/clexperiments.zip
    [3 ] http://people.rc.rit.edu/~bsm9339/corpora/szeged_uncertainty/merged_data
    """
    try:
        json_dict = json.loads(open(json_data, 'r').read())
        json_dict2 = {}
        cnt = 0
        for item in json_dict:
            cnt += 1
            if bool(item['ccue']):
                json_dict2.update({item['text'].lower(): item['ccue']})

        print("Found " + str(cnt) + " documents with uncertainty cues.")

        sents = Sentences.from_lineslist(_get_sentences(tagged_data))
        X, y = sents.get_data()

        with open(tagged_data + ".new", "w", newline='') as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter='\t')
            for i, sent in enumerate(X):
                s = _clean_sent(sent)
                tags = {}
                if s in json_dict2.keys():
                    tags = json_dict2[s]
                elif re.sub(r"(\d\s)-(\s\d)", r"\1-\2", s) in json_dict2.keys():
                    tags = json_dict2[re.sub(r"(\d\s)-(\s\d)", r"\1-\2", s)]
                elif re.sub(r"\s/\s", r"/", s) in json_dict2.keys():
                    tags = json_dict2[re.sub(r"\s/\s", r"/", s)]
                elif re.sub(r'\\u00b1', '±', s) in json_dict2.keys():
                    tags = json_dict2[re.sub(r'\\u00b1', '±', s)]
                elif re.sub(r'\s-\s', '-', s) in json_dict2.keys():
                    tags = json_dict2[re.sub(r'\s-\s', '-', s)]
                elif re.sub(r"\\", '', s) in json_dict2.keys():
                    tags = json_dict2[re.sub(r"\\", '', s)]
                else:
                    tags = {}

                rows = []
                sent_label, word_labels = _get_sent_label(tags, sent)
                prepend = "000"
                for j, word in enumerate(sent.get_words()):
                    if j > 999:
                        prepend=""
                    elif j > 99:
                        prepend="0"
                    elif j > 9:
                        prepend="00"

                    row = ['sent' + str(i) + "token" + prepend + str(j),
                           str(word.word), str(word.root), str(word.pos)]

                    row.append(word.binary_label) # Binary Label
                    row.append(word_labels[j]) # Multiclass Label

                    for k, v in word.get_features().items():
                        row.append(str(k) + ":" + str(v))

                    tsv_writer.writerow(row)
                tsv_writer.writerow([""])

    except Exception as e:
        extype, exvalue, extrace = sys.exc_info()
        traceback.print_exception(extype, exvalue, extrace)
        return False

    return True
