from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse.linalg import norm
import pandas as pd
from argparse import ArgumentParser


def cosine_similarities(raw_texts, code_descriptions, take_max=False, tfidf_mapper=None):
    if not take_max:
      code_descriptions = code_descriptions.groupby('code')['description'].agg(lambda x: ' '.join(x)).reset_index()
    if tfidf_mapper is None:
      tfidf_mapper = TfidfVectorizer()
      dictionary_vectors = tfidf_mapper.fit(code_descriptions.description)

    unique_codes = code_descriptions.reset_index().groupby('code')['index'].agg(lambda group: list(group)).reset_index()

    dictionary_vectors = tfidf_mapper.transform(code_descriptions.description)
    raw_texts_vectors = tfidf_mapper.transform(raw_texts)
    products = raw_texts_vectors.dot(dictionary_vectors.T)
    raw_norms = norm(raw_texts_vectors, axis=1)
    dictionary_norms = norm(dictionary_vectors, axis=1)
    raw_norms[raw_norms == 0] = 1.0
    dictionary_norms[dictionary_norms == 0] = 1.0
    similarity_features = products/np.expand_dims(raw_norms, axis=1)/np.expand_dims(dictionary_norms, axis=0)
    if take_max:
      max_similarity_features = np.zeros((len(raw_texts), unique_codes.shape[0]))
      for row_idx, row in unique_codes.iterrows():
        max_similarity_features[:, row_idx] = np.squeeze(np.max(similarity_features[:, row['index']], axis=1))
      similarity_features = max_similarity_features
    return similarity_features


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--texts')
  parser.add_argument('--vocabulary')
  parser.add_argument('--save_to')
  parser.add_argument('--take_max', action='store_true')
  args = parser.parse_args()

  with open(args.texts, encoding='latin-1') as input_stream:
    raw_texts = input_stream.readlines()
  vocabulary = pd.read_csv(args.vocabulary)
  similarity_features = cosine_similarities(raw_texts, vocabulary, take_max=args.take_max)
  np.savetxt(args.save_to, similarity_features)
