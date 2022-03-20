import os
import sys

import codecs
import argparse
import pickle
import collections
from utils import get_processing_word, is_dataset_tag, make_sure_path_exists
from fastNLP import Instance, DataSet, Vocabulary, Const

def expand(x):
    sent=["<sos>"]+x[1:]+["<eos>"]
    return [x+y for x,y in zip(sent[:-1],sent[1:])]
    
def read_file(filename, processing_word=get_processing_word(lowercase=False)):
    dataset = DataSet()
    niter=0
    with codecs.open(filename, "r", "utf-16") as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("-DOCSTART-"):
                if len(words) != 0:
                    assert len(words)>2
                    if niter==1:
                        print(words,tags)
                    niter += 1
                    dataset.append(Instance(ori_words=words[:-1], ori_tags=tags[:-1]))
                    words, tags = [], []
            else:
                word, tag = line.split()
                word = processing_word(word)
                words.append(word)
                tags.append(tag.lower())
                
    dataset.apply_field(lambda x: [x[0]], field_name='ori_words', new_field_name='task')   
    dataset.apply_field(lambda x: len(x), field_name='ori_tags', new_field_name='seq_len')   
    dataset.apply_field(lambda x: expand(x), field_name='ori_words', new_field_name="bi1")
    return dataset


parser = argparse.ArgumentParser()
parser.add_argument("--training-data", required=True, dest=