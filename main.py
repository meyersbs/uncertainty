#!/usr/bin/env python3

import argparse


def train(args):
    classifier = Classifier(binary=not args.multiclass)
    classifier.train(args.filepath)


def predict(args):
    classifier = Classifier(
            granularity=args.granularity, binary=not args.multiclass
        )
    print('{}: {}'.format(args.sentence, classifier.predict(args.sentence)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Command line access to the Linguistic Uncertainty '
                        'Classifier Interface (LUCI).'
        )
    subparsers = parser.add_subparsers(title='Commands', dest='command')
    subparsers.required = True

    parser_train = subparsers.add_parser(
            'train', help='Train uncertainty classifier.'
        )
    parser_train.add_argument(
            '-m', '--multiclass', action='store_true',
            help='When set, the response variable is considered multi-class. '
                 'Consequently, a multi-class classifier is trained.'
        )
    parser_train.add_argument(
            'filepath',
            help='The absolute path to a file containining the training data.'
        )
    parser_train.set_defaults(handler=train)

    parser_predict = subparsers.add_parser(
            'predict', help='Predict uncertainty of a sentence.'
        )
    parser_predict.add_argument(
            '-g', '--granularity', choices=['word', 'sentence'],
            default='word',
            help='The granularity at which the prediction must be made. '
                 'Default is word.'
        )
    parser_predict.add_argument(
            '-m', '--multiclass', action='store_true',
            help='When set, the response variable is considered multi-class. '
                 'Consequently, a multi-class classifier is used for '
                 'prediction.'
        )
    parser_predict.add_argument(
            'sentence',
            help='A sentence for which the uncertainty is to be predicted.'
        )
    parser_predict.set_defaults(handler=predict)
    args = parser.parse_args()

    from uncertainty.classifier import Classifier
    args.handler(args)
