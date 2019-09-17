import os
import re
from glob import glob


class ann_files_iterator(object):

    def __init__(self, directory):
        self.directory = directory
        all_files = []
        txt_files_pattern = os.path.join(directory, '*.txt')
        ann_files_pattern = os.path.join(directory, '*.ann')
        txt_files = [txt_file for txt_file in glob(txt_files_pattern)]
        ann_files = [ann_file for ann_file in glob(ann_files_pattern)]
        self.txt_files = sorted(txt_files)
        self.ann_files = sorted(ann_files)

    def __iter__(self):
        return iter(zip(self.txt_files, self.ann_files))


if __name__ == '__main__':
    for txt_file, ann_file in ann_files_iterator('../annotated_data/'):
        assert re.sub(r'.txt$', '', txt_file) == re.sub(r'.ann$', '', ann_file)
        assert os.path.isfile(txt_file)
        assert os.path.isfile(ann_file)
