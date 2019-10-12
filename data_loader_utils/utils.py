import numpy as np
import os
import csv
import sys
from nltk.tokenize import word_tokenize
import logging
import torch
from torch.utils.data import TensorDataset


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None, sim_feature=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.sim_feature = sim_feature


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, sim_feature, length):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sim_feature = sim_feature
        self.length = length


class DataProcessor(object):

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class Processor(DataProcessor):
    def __init__(self, data_dir):
        super(Processor, self).__init__()
        self.data_dir = data_dir
        self.vocab = []

    def get_train_examples(self):
        return self._create_examples("train")

    def get_dev_examples(self):
        return self._create_examples("test")

    def get_labels(self):
        if self.vocab: return self.vocab
        label_vocab_path = os.path.join(self.data_dir, 'label_vocab.txt')
        with open(label_vocab_path) as label_input_stream:
            self.vocab = [label.strip() for label in label_input_stream]
        return self.vocab

    def _create_examples(self, set_type):
        examples = []
        path_to_entities = os.path.join(self.data_dir, set_type + '_texts.txt')
        path_to_labels = os.path.join(self.data_dir, set_type + '_labels.txt')
        # currently not used
        path_to_sim_features = os.path.join(self.data_dir, set_type + '_sim_features.txt')
        sim_features = np.loadtxt(path_to_sim_features)
        example_idx = 0
        with open(path_to_entities, encoding='utf-8') as entity_input_stream, \
                open(path_to_labels, encoding='utf-8') as labels_input_stream:
            for entity, label, sim_feature in zip(entity_input_stream, labels_input_stream, sim_features):
                guid = "%s-%s" % (set_type, example_idx)
                examples.append(InputExample(guid=guid, text_a=entity.strip(), text_b=None,
                                             label=label.strip(), sim_feature=sim_feature))
                example_idx += 1
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, nltk_tokenizer=False):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        if nltk_tokenizer:
            tokens_a = word_tokenize(example.text_a)
        else:
            tokens_a = tokenizer.tokenize(example.text_a)

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        if not nltk_tokenizer:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        else:
            tokens = tokens_a
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        length = len(input_ids)
        input_mask = [1] * len(input_ids)
        if nltk_tokenizer:
            input_mask = [0] * len(input_ids)
            input_mask[len(input_ids) - 1] = 1
        # Zero-pad up to the sequence length.
        pad_size = (max_seq_length - len(input_ids))
        padding = [0] * (max_seq_length - len(input_ids))
        if nltk_tokenizer:
            input_ids = np.pad(input_ids, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
        else:
            input_ids += padding
        input_mask += padding
        segment_ids += padding

        if not nltk_tokenizer: assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        sim_feature = example.sim_feature

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          sim_feature=sim_feature,
                          length=length))
    return features


def load_build_dataset(args, tokenizer, evaluate=False):
    examples = Processor.get_dev_examples(args.data_dir) if evaluate else Processor.get_train_examples(args.data_dir)
    label_list = Processor.get_labels()
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset
