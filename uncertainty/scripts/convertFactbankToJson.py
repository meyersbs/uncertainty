from xml2json import xml2json
from collections import defaultdict

import json

json_dict = {}

with open('factbank.json', 'r') as f:
    json_dict = json.loads(f.read())

def parse_sent(sent, prefix):
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

def parse_doc_dict(doc):
    if 'Sentence' in doc.keys():
        if type(doc['Sentence']) == list:
            for sent in doc['Sentence']:
                if type(sent) == dict:
                    all_sents.append(parse_sent(sent, ''))
        elif type(doc['Sentence']) == dict:
            prefix = ''
            if '#text' in doc['Sentence'].keys():
                prefix = doc['Sentence']['#text']
            if 'ccue' in doc['Sentence'].keys():
                all_sents.append(parse_sent(doc['Sentence'], prefix))
            else:
                all_sents.append({'text': prefix, 'ccue': {}})

all_sents = []
for document in json_dict['Annotation']['DocumentSet']['Document']:
    if type(document['DocumentPart']) == list:
        for doc in document['DocumentPart']:
            if type(doc) == dict:
                parse_doc_dict(doc)
    elif type(document['DocumentPart']) == dict:
        doc = document['DocumentPart']
        parse_doc_dict(doc)
    else:
        print(type(document['DocumentPart']))
#print("=====")
#print(len(all_sents))
#print(all_sents[10:50])


with open('factbank.cleaned.json', 'w+') as f:
     f.write(json.dumps(all_sents))

