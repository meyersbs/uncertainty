import json
import subprocess

from xml2json import xml2json
from collections import defaultdict


class Converter(object):
    def __init__(self, old, raw, new):
        self.JSON_DICT = {}
        self.ALL_SENTS = []
        self.OLD = old
        self.RAW = raw
        self.NEW = new

    def _xmlToJson(self):
        bash = str("xml2json -t xml2json -o " + self.RAW + " " + self.OLD)
        process = subprocess.call(bash.split())

    def _load_raw(self):
        with open(self.RAW, 'r') as f:
            self.JSON_DICT = json.loads(f.read())

    def _parse_sent(self, sent):
        raise NotImplementError

    def _convert(self):
        raise NotImplementedError

    def _write_to_file(self):
        with open(self.NEW, 'w') as f:
            f.write(json.dumps(self.ALL_SENTS))

    def convert(self):
        self._xmlToJson()
        self._load_raw()
        self._convert()
        self._write_to_file()


class WikiConverter(Converter):

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


class FactbankConverter(Converter):

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


class BioBmcConverter(WikiConverter):{}


class BioFlyConverter(WikiConverter):{}


class BioHbcConverter(FactbankConverter):{}
