
import argparse
import pickle
import collections
import logging
import math
import os,sys,time
import random
from sys import maxsize
import pickle
import numpy as np
import torch
import torch.nn as nn
import fastNLP
from fastNLP import BucketSampler,SequentialSampler
from fastNLP import DataSetIter
import optm
import models
import utils


NONE_TAG = "<NONE>"
START_TAG = "<sos>"
END_TAG = "<eos>"

DEFAULT_WORD_EMBEDDING_SIZE = 100
DEBUG_SCALE = 200

# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
parser.add_argument("--word-embeddings", dest="word_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--bigram-embeddings", dest="bigram_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--crf", dest="crf", action="store_true", help="whether to use CRF")                    
parser.add_argument("--devi", default="0", dest="devi", help="gpu to use")
parser.add_argument("--step", default=0, dest="step", type=int,help="step")
parser.add_argument("--num-epochs", default=80, dest="num_epochs", type=int,
                    help="Number of epochs through training set")
parser.add_argument("--flex", default=-1, dest="flex", type=int,
                    help="Number of epochs through training set after freezing the pretrained embeddings")
parser.add_argument("--batch-size", default=256, dest="batch_size", type=int,
                    help="Minibatch size of training set")
parser.add_argument("--d_model", default=256, dest="d_model", type=int, help="d_model of transformer encoder")
parser.add_argument("--d_ff", default=1024, dest="d_ff", type=int, help="d_ff for FFN")
parser.add_argument("--N", default=6, dest="N", type=int, help="Number of layers")
parser.add_argument("--h", default=4, dest="h", type=int, help="Number of head")
parser.add_argument("--factor", default=2, dest="factor", type=float, help="factor for learning rate")
parser.add_argument("--dropout", default=0.2, dest="dropout", type=float,
                    help="Amount of dropout(not keep rate, but drop rate) to apply to embeddings part of graph")
parser.add_argument("--log-dir", default="result", dest="log_dir",
                    help="Directory where to write logs / saved models")
parser.add_argument("--task-name", default=time.strftime("%Y-%m-%d-%H-%M-%S"), dest="task_name",
                    help="Name for this task, use a comprehensive one")
parser.add_argument("--no-model", dest="no_model", action="store_true", help="Don't save model")
parser.add_argument("--always-model", dest="always_model", action="store_true",
                    help="Always save the model after every epoch")
parser.add_argument("--old-model", dest="old_model", help="Path to old model for incremental training")
parser.add_argument("--skip-dev", dest="skip_dev", action="store_true", help="Skip dev set during training")
parser.add_argument("--freeze", dest="freeze", action="store_true", help="freeze pretrained embeddings")
parser.add_argument("--only-task", dest="only_task", action="store_true", help="only train task embeddings")
parser.add_argument("--subset", dest="subset", help="Only train and test on a subset of the whole dataset")
parser.add_argument("--seclude", dest="seclude", help="train and test except a subset of the copora")
parser.add_argument("--instances", default=None, dest="instances", type=int,help="num of instances of subset")

parser.add_argument("--python-seed", dest="python_seed", type=int, default=random.randrange(maxsize),
                    help="Random seed of Python and NumPy")
parser.add_argument("--debug", dest="debug", default=False, action="store_true", help="Debug mode")
parser.add_argument("--test", dest="test", action="store_true", help="Test mode")

options = parser.parse_args()
task_name = options.task_name
root_dir = "{}/{}".format(options.log_dir, task_name)
utils.make_sure_path_exists(root_dir)

devices=[int(x) for x in options.devi]
device = torch.device("cuda:{}".format(devices[0]))  

def init_logger():
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    log_formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    file_handler = logging.FileHandler("{0}/info.log".format(root_dir), mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


# ===-----------------------------------------------------------------------===
# Set up logging
# ===-----------------------------------------------------------------------===
logger = init_logger()

# ===-----------------------------------------------------------------------===
# Log some stuff about this run
# ===-----------------------------------------------------------------------===
logger.info(' '.join(sys.argv))
logger.info('')