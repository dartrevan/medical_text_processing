from model import build_model
from argparse import ArgumentParser
import numpy as np
import os
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from keras.callbacks import CSVLogger


def load_data(data_directory, ds_type):
    med_entities_path = os.path.join(data_directory, ds_type + '_med_entities.txt')
    labels_path = os.path.join(data_directory, ds_type + '_labels.txt')

    with open(med_entities_path, encoding='latin-1') as input_stream:
        texts = [line.strip().lower() for line in input_stream]

    seq_lengths = [len(text) for text in texts]

    with open(labels_path, encoding='latin-1') as input_stream:
        labels = [line.strip() for line in input_stream]

    return texts, labels, seq_lengths


def token2ids(texts, vocab={}):
    texts_ids = []
    for text in texts:
        text_ids = []
        for token in word_tokenize(text):
            if token not in vocab: vocab[token] = len(vocab) + 1
            text_ids.append(vocab[token])
        texts_ids.append(text_ids)
    return texts_ids, vocab


def get_embeddings(w2v_model, vocab):
    embeddings_matrix = np.zeros((len(vocab) + 1, 200))
    for token, token_id in vocab.items():
        if token in w2v_model:
            embeddings_matrix[token_id] = w2v_model[token]
    return embeddings_matrix


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--log_file')
    parser.add_argument('--emb_file', default=None)
    parser.add_argument('--use_elmo_embeddings', action='store_true')
    args = parser.parse_args()

    train_texts, train_labels, train_lengths = load_data(args.data_dir, 'train')
    test_texts, test_labels, test_lengths = load_data(args.data_dir, 'test')

    train_texts, vocab = token2ids(train_texts)
    test_texts, vocab = token2ids(test_texts, vocab)
    if args.emb_file is not None:
        w2v_model = KeyedVectors.load_word2vec_format(args.emb_file, binary=True)
        embeddings_matrix = get_embeddings(w2v_model, vocab)
    else:
        embeddings_matrix = None

    train_texts = pad_sequences(train_texts, maxlen=30, dtype='int32', padding='post', value=0)
    test_texts = pad_sequences(test_texts, maxlen=30, dtype='int32', padding='post', value=0)

    train_texts = np.array(train_texts, dtype=object)
    test_texts = np.array(test_texts, dtype=object)
    labels = set(train_labels + test_labels)
    label2id = {label.strip(): i for i, label in enumerate(labels)}
    train_labels = [label2id[label] for label in train_labels]
    test_labels = [label2id[label] for label in test_labels]

    model = build_model(num_labels=len(label2id), max_length=30, embedding_matrix=embeddings_matrix,
                        elmo_embeddings=args.use_elmo_embeddings)
    csv_logger = CSVLogger(args.log_file)
    model.fit(train_texts, train_labels, validation_data=(test_texts, test_labels), epochs=50, batch_size=32,
              callbacks=[csv_logger])
