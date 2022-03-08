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
                   