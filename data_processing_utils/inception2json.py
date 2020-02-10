from argparse import ArgumentParser
import pandas as pd
from glob import glob
import json
import os


def read_texts(file_path):
    texts = []
    with open(file_path, encoding='utf-8') as input_stream:
        for line in input_stream:
            if not line.startswith('#Text='): continue
            texts.append(line[6:].strip())
    return texts


def get_text_from_spans(entity, sentences):
    sentence = sentences[entity['sentence_id']]
    return sentence[entity['start']:entity['end']]


def get_entities_spans(annotation_data):
    annotation_data['sentence_id'] = annotation_data.token_id.apply(lambda tid: int(tid.split('-')[0]))
    annotation_data['entity_type'] = annotation_data.entity_type.apply(lambda eid: eid[:-3])
    annotation_data['start'] = annotation_data.token_span.apply(lambda tid: int(tid.split('-')[0]))
    annotation_data['end'] = annotation_data.token_span.apply(lambda tid: int(tid.split('-')[1]))
    sentences_starts = annotation_data.groupby('sentence_id')['start'].min().reset_index().\
        rename(columns={'start': 'sentence_start'})

    annotation_data = annotation_data[annotation_data.entity_id != '_']
    entities_starts = annotation_data.groupby(['sentence_id', 'entity_id', 'entity_type'])['start'].min().reset_index()
    entities_ends = annotation_data.groupby(['sentence_id', 'entity_id', 'entity_type'])['end'].max().reset_index()

    entities = pd.merge(entities_starts, entities_ends, on=['sentence_id', 'entity_id', 'entity_type'])
    entities = pd.merge(entities, sentences_starts, on=['sentence_id'])
    entities['start'] = entities['start'] - entities['sentence_start']
    entities['end'] = entities['end'] - entities['sentence_start']
    return entities


def parse_sentences(file_path):
    resulting_data = []
    sentences = read_texts(file_path)
    annotation_data = pd.read_csv(file_path, sep='\t', skip_blank_lines=True, comment='#', encoding='utf-8',
                                  names=['token_id', 'token_span', 'token', 'entity_id', 'entity_type',
                                         'unknown_1', 'unknown_2'])
    entities_spans = get_entities_spans(annotation_data)
    entities_spans['entity_text'] = entities_spans.apply(get_text_from_spans, args=(sentences,), axis=1)
    for sentence_id, sentence in enumerate(sentences):
        sentence_annotations = entities_spans[entities_spans.sentence_id == sentence_id]
        sentence_annotations = sentence_annotations[['entity_id', 'entity_type', 'start', 'end', 'entity_text']]
        resulting_data.append({
            'text': sentence,
            'sentence_id': sentence_id,
            'file_name': os.path.basename(file_path),
            'entities': sentence_annotations.to_dict(orient='records')
        })
    return resulting_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_folder')
    parser.add_argument('--save_to')
    args = parser.parse_args()

    output_data = []
    file_pattern = os.path.join(args.input_folder, '*')
    for file_path in glob(file_pattern):
        sentences = parse_sentences(file_path)
        output_data += sentences

    with open(args.save_to, 'w', encoding='utf-8') as output_stream:
        for document in output_data:
            serialized_data = json.dumps(document, ensure_ascii=False)
            output_stream.write(serialized_data + '\n')

