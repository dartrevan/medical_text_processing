from __future__ import absolute_import, division, print_function
from allennlp.modules.elmo import batch_to_ids

from nltk.tokenize import word_tokenize
import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
#import torchcontrib

from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score

from simple_classifier import BertForSequenceClassification, ElmoSequenceClassification
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

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
        path_to_entities = os.path.join(self.data_dir, set_type + '_med_entities.txt')
        path_to_labels = os.path.join(self.data_dir, set_type + '_labels.txt')
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


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, elmo):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        if elmo:
            tokens_a = word_tokenize(example.text_a)
        else:
            tokens_a = tokenizer.tokenize(example.text_a)

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        if not elmo:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        else:
            tokens = tokens_a
        segment_ids = [0] * len(tokens)

        if elmo:
            input_ids = batch_to_ids([tokens]).numpy()[0]
        else:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
        length = len(input_ids)
        input_mask = [1] * len(input_ids)
        if elmo:
            input_mask = [0] * len(input_ids)
            input_mask[len(input_ids) - 1] = 1
        # Zero-pad up to the sequence length.
        pad_size = (max_seq_length - len(input_ids))
        padding = [0] * (max_seq_length - len(input_ids))
        if elmo:
            input_ids = np.pad(input_ids, ((0, pad_size), (0,0)), mode='constant', constant_values=0)
        else:
            input_ids += padding
        input_mask += padding
        segment_ids += padding

        if not elmo: assert len(input_ids) == max_seq_length
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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {'acc': simple_accuracy(preds, labels)}


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--elmo",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--use_sims",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    processor = Processor(args.data_dir)

    label_list = processor.get_labels()
    num_labels = len(label_list)
    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    if args.elmo:
        model = ElmoSequenceClassification(None, num_labels)
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                              cache_dir=cache_dir,
                                                              num_labels=num_labels,
                                                              use_sims=args.use_sims)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        if args.elmo:
            model = ElmoSequenceClassification(None, num_labels)
        else:
            config = BertConfig(output_config_file)
            model = BertForSequenceClassification(config, num_labels=num_labels, use_sims=args.use_sims)
        model.load_state_dict(torch.load(output_model_file))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0 # None
    if args.do_train:
        train_examples = processor.get_train_examples()
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)
    #oprimizer_bert = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    #optimizer =  torchcontrib.optim.SWA(optimizer_bert, swa_start=10, swa_freq=5, swa_lr=0.05)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, args.elmo)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        # print(np.concatenate([f.segment_ids for f in train_features], axis=-1).shape)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_sim_features = torch.tensor([f.sim_feature for f in train_features], dtype=torch.float)

        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sim_features, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, sim_features, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None, sim_features=sim_features)

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
        #optimizer.swap_swa_sgd()
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        if not args.elmo:
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())

        # Load a trained model and config that you have fine-tuned
        if args.elmo:
            pass
            # model = ElmoSequenceClassification(None, num_labels)
        else:
            config = BertConfig(output_config_file)
            model = BertForSequenceClassification(config, num_labels=num_labels, use_sims=args.use_sims)
        model.load_state_dict(torch.load(output_model_file))
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels, use_sims=args.use_sims)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples()
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, args.elmo)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_sim_features = torch.tensor([f.sim_feature for f in eval_features], dtype=torch.float)

        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sim_features, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, sim_features, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            sim_features = sim_features.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None, sim_features=sim_features)

            # create eval loss and other metric required by the task

            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        preds = np.argmax(preds, axis=1)
        preds_mapped = [label_list[pred] for pred in preds]
        output_predictions = os.path.join(args.output_dir, 'predictions.txt')
        with open(output_predictions, 'w') as output_stream:
            for pred in preds_mapped:
                output_stream.write(pred + '\n')
        result = compute_metrics(preds, all_label_ids.numpy())
        loss = tr_loss / nb_tr_steps if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
