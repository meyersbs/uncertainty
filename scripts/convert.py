import json
import subprocess

from xml2json import xml2json
from collections import defaultdict

from corpora import *


class Converter(object):
    def __init__(self):
        self.JSON_DICT = {}
        self.ALL_SENTS = []

    def _xmlToJson(self, xml_filename, json_filename):
        bash = str("xml2json -t xml2json -o " + json_filename + " " + xml_filename)
        process = subprocess.call(bash.split())

    def _load_raw(self, json_filename):
        with open(json_filename, 'r') as f:
            self.JSON_DICT = json.loads(f.read())

    def _parse_sent(self, sent):
        raise NotImplementError

    def _convert(self):
        raise NotImplementedError

    def _write_to_file(self, json_filename):
        with open(json_filename, 'w') as f:
            f.write(json.dumps(self.ALL_SENTS))

    def convert(self):
        raise NotImplementedError


class WikiConverter(Converter):
    from corpora import WIKI_OLD, WIKI_RAW, WIKI_NEW

    def _parse_sent(self, sent):
        results = {'text': "", 'ccue': {}}
        curr_sent = ""
        curr_types = defaultdict(list)
        if '#text' in sent.keys():
            curr_sent += sent['#text']
        if 'ccue' in sent.keys():
            ccue = sent['ccue']
            if type(ccue) == dict:
                curr_sent += ccue['#text'] + ccue['#tail']
                curr_types[ccue['@type']].append(ccue['#text'])
            elif type(ccue) == list:
                for sub_ccue in ccue:
                    curr_sent += sub_ccue['#text'] + sub_ccue['#tail']
                    curr_types[sub_ccue['@type']].append(sub_ccue['#text'])
        results['text'] = curr_sent
        results['ccue'] = dict(curr_types)
        return results

    def _convert(self):
        for document in self.JSON_DICT['Annotation']['DocumentSet']['Document']:
            for doc in document['DocumentPart']:
                if type(doc['Sentence']) == list:
                    for sent in doc['Sentence']:
                        if type(sent) == dict:
                            self.ALL_SENTS.append(self._parse_sent(sent))
                elif type(doc['Sentence']) == dict:
                    if '#text' in doc['Sentence'].keys():
                        self.ALL_SENTS.append({'text': doc['Sentence']['#text'], 'ccue': {}})

    def convert(self):
        global WIKI_OLD, WIKI_RAW, WIKI_NEW
        self._xmlToJson(WIKI_OLD, WIKI_RAW)
        self._load_raw(WIKI_RAW)
        self._convert()
        self._write_to_file(WIKI_NEW)


class FactbankConverter(Converter):
    from corpora import FACTBANK_OLD, FACTBANK_RAW, FACTBANK_NEW

    def _parse_sent(self, sent, prefix):
        results = {'text': "", 'ccue': {}}
        curr_sent = prefix
        curr_types = defaultdict(list)
        if '#text' in sent.keys():
            curr_sent += sent['#text']
        if 'ccue' in sent.keys():
            ccue = sent['ccue']
            if type(ccue) == dict:
                curr_sent += ccue['#text'] + ccue['#tail']
                curr_types[ccue['@type']].append(ccue['#text'])
            elif type(ccue) == list:
                for sub_ccue in ccue:
                    curr_sent += sub_ccue['#text'] + sub_ccue['#tail']
                    curr_types[sub_ccue['@type']].append(sub_ccue['#text'])
        results['text'] = curr_sent
        results['ccue'] = dict(curr_types)
        return results

    def _parse_doc(self, doc):
        if 'Sentence' in doc.keys():
            if type(doc['Sentence']) == list:
                for sent in doc['Sentence']:
                    if type(sent) == dict:
                        self.ALL_SENTS.append(self._parse_sent(sent, ''))
            elif type(doc['Sentence']) == dict:
                prefix = ''
                if '#text' in doc['Sentence'].keys():
                    prefix = doc['Sentence']['#text']
                if 'ccue' in doc['Sentence'].keys():
                    self.ALL_SENTS.append(self._parse_sent(doc['Sentence'], prefix))
                else:
                    self.ALL_SENTS.append({'text': prefix, 'ccue': {}})

    def _convert(self):
        for document in self.JSON_DICT['Annotation']['DocumentSet']['Document']:
            if type(document['DocumentPart']) == list:
                for doc in document['DocumentPart']:
                    if type(doc) == dict:
                        self._parse_doc(doc)
            elif type(document['DocumentPart']) == dict:
                doc = document['DocumentPart']
                self._parse_doc(doc)
            else:
                print(type(document['DocumentPart']))

    def convert(self):
        global FACTBANK_OLD, FACTBANK_RAW, FACTBANK_NEW
        self._xmlToJson(FACTBANK_OLD, FACTBANK_RAW)
        self._load_raw(FACTBANK_RAW)
        self._convert()
        self._write_to_file(FACTBANK_NEW)
