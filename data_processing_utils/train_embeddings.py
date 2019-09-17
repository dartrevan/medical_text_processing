from gensim.models import Word2Vec
from gensim.models.word2vec import  LineSentence
from argparse import ArgumentParser
import json
import os
import logging
import re


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class SentencesIterator():
  def __init__(self, directory, year):
    # self.input_files_paths = [directory]
    self.input_files_paths = []
    for input_file in os.listdir(directory):
      fyear = int(re.findall('\d+', input_file)[-1])
      if fyear > year: continue
      self.input_files_paths += [os.path.join(directory, input_file)]

  def __iter__(self):
    self.current_file_id = -1
    self.sentences = []
    self.current_sentence_id = 0
    return self

  def read_file(self, file_id):
    current_input_file_path = self.input_files_paths[file_id]
    with open(current_input_file_path) as input_stream:
      self.sentences = [json.loads(line) for line in input_stream]
    self.current_sentence_id = -1

  def __next__(self):
    if len(self.sentences) - 1 <= self.current_sentence_id:
      self.current_file_id += 1
      if self.current_file_id >= len(self.input_files_paths): raise StopIteration()
      self.read_file(self.current_file_id)
    self.current_sentence_id += 1
    return self.sentences[self.current_sentence_id]


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--input_dir', default=None)
  parser.add_argument('--input_file', default=None)
  parser.add_argument('--save_to')
  parser.add_argument('--year', type=int)
  parser.add_argument('--size', type=int, default=250)
  parser.add_argument('--min_count', type=int, default=10)
  parser.add_argument('--window', type=int, default=10)
  parser.add_argument('--negative', type=int, default=15)
  args = parser.parse_args()

  if args.input_dir: sentences = SentencesIterator(args.input_dir, args.year)
  if args.input_file: sentences =  LineSentence(args.input_file)
  w2v_model = Word2Vec(sentences, size=args.size, min_count=args.min_count, window=args.window, negative=args.negative, sg=1, workers=1)
  w2v_model.wv.save_word2vec_format(args.save_to)
