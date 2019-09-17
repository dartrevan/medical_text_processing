from utils import read_json_file
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from argparse import ArgumentParser

END_OF_SENT = '\n'


def get_document_parts(text, by_sent=False):
    if not by_sent: return [(0, len(text))]
    return list(PunktSentenceTokenizer().span_tokenize(text))


def get_bio_tag(token_idx, entity):
    if entity['type'] == 'Medication' and ('MedType' not in entity or entity['MedType'] != 'Drugname'): return 'O'
    if entity['type'] == 'O': return 'O'
    bio_tag = 'I-'
    if token_idx == 0: bio_tag = 'B-'
    return bio_tag + entity['type']


def get_token_bio_tags(text, entity):
    conll_output = []
    for i, token in enumerate(wordpunct_tokenize(text)):
        bio_tag = get_bio_tag(i, entity)
        conll_output.append([token, bio_tag])
    return conll_output


def process_tail(document_parts, offset, entity_start, text, current_document_part_id):
    conll_output = []
    current_document_part_start, current_document_part_end = document_parts[current_document_part_id]
    while current_document_part_end < entity_start:
        offset = max(current_document_part_start, offset)
        current_document_part = text[offset:current_document_part_end]
        conll_output += get_token_bio_tags(current_document_part, {'type':'O'})
        conll_output.append(['', '\n'])
        current_document_part_id += 1
        if len(document_parts) <= current_document_part_id:
            break
        current_document_part_start, current_document_part_end = document_parts[current_document_part_id]

    return conll_output, current_document_part_id


def to_conll(document, by_sent=False):
    conll_output = []
    prev_entity_end = 0
    entities = sorted(document['entities'].values(), key=lambda entity: entity['start'])
    document_parts = get_document_parts(document['text'], by_sent)
    current_document_part_id = 0
    current_document_part_start, current_document_part_end = document_parts[current_document_part_id]
    for entity in entities:
        entity_start = entity['start']
        entity_end = entity['end']
        if current_document_part_end < entity_start:
            tail_conll, current_document_part_id = process_tail(document_parts, prev_entity_end, entity_start,
                                                                  document['text'], current_document_part_id)
            current_document_part_start, current_document_part_end = document_parts[current_document_part_id]
            conll_output += tail_conll

        offset = max(prev_entity_end, current_document_part_start)
        no_entity_part = document['text'][offset:entity_start]
        conll_output += get_token_bio_tags(no_entity_part, {'type':'O'})
        entity_part = document['text'][entity_start:entity_end]
        conll_output += get_token_bio_tags(entity_part, entity)
        prev_entity_end = entity_end

    tail_conll, current_document_part_id = process_tail(document_parts, prev_entity_end, len(document['text']) + 1,
                                                        document['text'], current_document_part_id)
    conll_output += tail_conll
    return conll_output


def format_output_line(token, tag):
    if token:
        return '{} {}\n'.format(token, tag)
    return '\n'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--save_to', type=str, help='')
    parser.add_argument('--save_document_ids_to', type=str, help='')
    parser.add_argument('--jsonl', action='store_true', help='')
    parser.add_argument('--by_sent', action='store_true', help='')
    args = parser.parse_args()

    data = read_json_file(args.input, args.jsonl)

    # TODO: write to_conll as iterator by sentences
    with open(args.save_to, 'w', encoding='utf-8') as conll_output_stream, \
            open(args.save_document_ids_to, 'w', encoding='utf-8') as document_ids_output_stream:
        for document in data:
            if document['text'] == '':
                print('Warning empty document')
                continue
            for token, bio_tag in to_conll(document, args.by_sent):
                output_line = format_output_line(token, bio_tag)
                conll_output_stream.write(output_line)
                if output_line == END_OF_SENT:
                    document_ids_output_stream.write(document['document_id'] + '\n')

