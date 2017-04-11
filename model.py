import sys
import os
import _pickle

from scripts.convert import *
from features.features import *
from corpora import *

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 1:
        sys.exit(1)
    else:
        if args[0] == "wiki":
            WikiConverter(WIKI_OLD, WIKI_RAW, WIKI_NEW).convert()
        elif args[0] == "factbank":
            FactbankConverter(FACTBANK_OLD, FACTBANK_RAW, FACTBANK_NEW).convert()
        elif args[0] == "biobmc":
            BioBmcConverter(BIO_BMC_OLD, BIO_BMC_RAW, BIO_BMC_NEW).convert()
        elif args[0] == "biofly":
            BioFlyConverter(BIO_FLY_OLD, BIO_FLY_RAW, BIO_FLY_NEW).convert()
        elif args[0] == "biohbc":
            BioHbcConverter(BIO_HBC_OLD, BIO_HBC_RAW, BIO_HBC_NEW).convert()
        elif args[0] == "features":
            with open(WIKI_NEW, 'r') as f:
                sents = json.loads(f.read())
                print(features(sents))
