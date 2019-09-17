import pandas as pd
import json
from nltk.tokenize import PunktSentenceTokenizer
from argparse import ArgumentParser

sent_tokenizer = PunktSentenceTokenizer()


def split_sentences(document):
    sentences = []
    entity_id = 0
    for sentence_start, sentence_end in sent_tokenizer.span_tokenize(document['text']):
        sentence = document['text'][sentence_start:sentence_end]
        if len(document['annotations']) <= entity_id:
            sentences.append({
                'sentence': sentence,
                'entity': None,
                'start': None,
                'end': None,
                'concepts': 'CONCEPT_LESS'
            })
            break
        entity = document['annotations'][entity_id]
        entity_start, entity_end = entity['span']
        if entity_start > sentence_end:
            sentences.append({
                'sentence': sentence,
                'entity': None,
                'start': None,
                'end': None,
                'concepts': 'CONCEPT_LESS'
            })
        while entity_start < sentence_end and entity_id < len(document['annotations']):
            entity = document['annotations'][entity_id]
            entity_start, entity_end = entity['span']
            sentences.append({
                'sentence': sentence,
                'entity': entity['text'],
                'start': str(entity_start),
                'end': str(entity_end),
                'concepts': ','.join([concept['concept_id'] for concept in entity['concepts']])
            })
            entity_id += 1
    return sentences


if __name__ == '__main__':
    parser = ArgumentParser("Scripts for converting json file into conll formatted file")
    parser.add_argument('--input', type=str, help="Input file in json format")
    parser.add_argument('--save_to', type=str, help="Output file in tsv format")
    args = parser.parse_args()

    with open(args.input, encoding='utf-8') as input_stream:
        data = json.loads(input_stream.read())
    data = data.values()
    entities = [sentence for document in data for sentence in split_sentences(document)]
    pd.DataFrame(entities)[['sentence', 'entity', 'start', 'end', 'concepts']].to_csv(args.save_to, encoding='utf-8',
                                                                                      index=False, sep='\t')
