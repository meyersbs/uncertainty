import sys
import os
import _pickle

from scripts.convert import *

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 1:
        sys.exit(1)
    else:
        if args[0] == "wiki":
            WikiConverter().convert()
        elif args[0] == "factbank":
            FactbankConverter().convert()
