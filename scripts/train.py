import _pickle
import json
import numpy as np
import os
import random
import sys

import features as f

def is_hedge(ngram, size):
    if size == 3:
        if " ".join(ngram) in TRI_HEDGES:
            return True
    elif size == 2:
        if " ".join(ngram) in BI_HEDGES:
            return True
    elif size == 1:
        if ngram in UNI_HEDGES:
            return True
    else:
        raise ValueError("Invalid value for 'size': " + str(size))

    return False

def is_weasel(ngram, size):
    if size == 3:
        if " ".join(ngram) in TRI_WEASELS:
            return True
    elif size == 2:
        if " ".join(ngram) in BI_WEASELS:
            return True
    elif size == 1:
        if ngram in UNI_WEASELS:
            return True
    else:
        raise ValueError("Invalid value for 'size': " + str(size))

    return False

def is_peacock(ngram, size):
    if size == 3:
        if " ".join(ngram) in TRI_PEACOCKS:
            return True
    elif size == 2:
        if " ".join(ngram) in BI_PEACOCKS:
            return True
    elif size == 1:
        if ngram in UNI_PEACOCKS:
            return True
    else:
        raise ValueError("Invalid value for 'size': " + str(size))

    return False
