from argparse import ArgumentParser
import pandas as pd
import json


def read_texts(file_path):
    texts = []
    with open(file_path, encoding='utf-8') as input_stream:
        for line in input_stream:
            if not line.startswith('#Text='): continue
            texts.append(line[6:])
    return texts


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_csv')
    parser.add_argument('--save_to')
    args = parser.parse_args()
    annotation_data = pd.read_csv(args.input_csv, sep='\t', skip_blank_lines=True, comment='#', encoding='utf-8',
                                  names=['token_id', 'token_span', 'token', 'entity_id', 'entity_type',
                                         'unknown_1', 'unknown_2'])
    texts = read_texts(args.input_csv)
    annotation_data = annotation_data[annotation_data.entity_id != '_']
    annotation_data['text_id'] = annotation_data.token_id.apply(lambda tid: int(tid.split('-')[0]))
    annotation_data['entity_type'] = annotation_data.entity_type.apply(lambda eid: eid[:-3])
    annotation_data['start'] = annotation_data.token_span.apply(lambda tid: int(tid.split('-')[0]))
    annotation_data['end'] = annotation_data.token_span.apply(lambda tid: int(tid.split('-')[1]))
    entities_starts = annotation_data.groupby(['text_id', 'entity_id', 'entity_type'])['start'].min().reset_index()
    entities_ends = annotation_data.groupby(['text_id', 'entity_id', 'entity_type'])['end'].max().reset_index()
    entities = pd.merge(entities_starts, entities_ends, on=['text_id', 'entity_id', 'entity_type'])
    entities['entity_text'] = entities.apply(lambda entity:
                                              texts[entity['text_id']][entity['start']:entity['end']], axis=1)
    output_data = []
    for text_id, text in enumerate(texts):
        text_annotations = entities[entities.text_id == text_id]
        text_annotations = text_annotations[['entity_id', 'entity_type', 'start', 'end', 'entity_text']]
        output_data.append({
            'text': text,
            'entities': text_annotations.to_dict(orient='records')
        })

    with open(args.save_to, 'w', encoding='utf-8') as output_stream:
        for document in output_data:
            serialized_data = json.dumps(document)
            output_stream.write(serialized_data + '\n')

