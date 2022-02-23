
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
logger.info(options)

if options.debug:
    print("DEBUG MODE")
    options.num_epochs = 2
    options.batch_size=20

random.seed(options.python_seed)
np.random.seed(options.python_seed % (2 ** 32 - 1))
logger.info('Python random seed: {}'.format(options.python_seed))

# ===-----------------------------------------------------------------------===
# Read in dataset
# ===-----------------------------------------------------------------------===
dataset = pickle.load(open(options.dataset, "rb"))
train_set=dataset["train_set"]
test_set=dataset["test_set"]
uni_vocab=dataset["uni_vocab"]
bi_vocab=dataset["bi_vocab"]
task_vocab=dataset["task_vocab"]
tag_vocab=dataset["tag_vocab"]
print(bi_vocab.to_word(0),tag_vocab.word2idx)
print(task_vocab.word2idx)
if options.skip_dev:
    dev_set=test_set
else:
    train_set, dev_set=train_set.split(0.1)
    
print(len(train_set),len(dev_set),len(test_set))

if options.debug:
    train_set = train_set[0:DEBUG_SCALE]
    dev_set = dev_set[0:DEBUG_SCALE]
    test_set = test_set[0:DEBUG_SCALE]

# ===-----------------------------------------------------------------------===
# Build model and trainer
# ===-----------------------------------------------------------------------===

if options.word_embeddings is None:
    init_embedding=None
else:
    print("Load:",options.word_embeddings)
    init_embedding=fastNLP.io.embed_loader.EmbedLoader.load_with_vocab(options.word_embeddings, uni_vocab, normalize=False)
    
bigram_embedding = None
if options.bigram_embeddings:
    if options.bigram_embeddings == 'merged':
        logging.info('calculate bigram embeddings from unigram embeddings')
        bigram_embedding=np.random.randn(len(bi_vocab), init_embedding.shape[-1]).astype('float32')      
        for token, i in bi_vocab:
            if token.startswith('<') and token.endswith('>'): continue
            if token.endswith('>'):
                x,y=uni_vocab[token[0]], uni_vocab[token[1:]]
            else: 
                x,y=uni_vocab[token[:-1]], uni_vocab[token[-1]]
            if x==uni_vocab['<unk>']:
                x=uni_vocab['<pad>']
            if y==uni_vocab['<unk>']:
                y=uni_vocab['<pad>']
            bigram_embedding[i]=(init_embedding[x]+init_embedding[y])/2
    else:    
        print("Load:",options.bigram_embeddings)
        bigram_embedding=fastNLP.io.embed_loader.EmbedLoader.load_with_vocab(options.bigram_embeddings, bi_vocab, normalize=False)

#select subset training
if options.seclude is not None:
    setname="<{}>".format(options.seclude)
    print("seclude",setname)
    train_set.drop(lambda x: x["ori_words"][0]==setname,inplace=True)
    test_set.drop(lambda x: x["ori_words"][0]==setname,inplace=True)
    dev_set.drop(lambda x: x["ori_words"][0]==setname,inplace=True)

if options.subset is not None:
    setname="<{}>".format(options.subset)
    print("select",setname)
    train_set.drop(lambda x: x["ori_words"][0]!=setname,inplace=True)
    test_set.drop(lambda x: x["ori_words"][0]!=setname,inplace=True)
    dev_set.drop(lambda x: x["ori_words"][0]!=setname,inplace=True)
    if options.instances is not None:
        train_set=train_set[:int(options.instances)]
        
# build model and optimizer    
i2t=None
if options.crf:
    #i2t=utils.to_id_list(tag_vocab.word2idx)   
    i2t={}
    for x,y in tag_vocab.word2idx.items():
        i2t[y]=x
    print("use crf:",i2t)

freeze=True if options.freeze else False
model = models.make_CWS(d_model=options.d_model, N=options.N, h=options.h, d_ff=options.d_ff,dropout=options.dropout,word_embedding=init_embedding,bigram_embedding=bigram_embedding,tag_size=len(tag_vocab),task_size=len(task_vocab),crf=i2t,freeze=freeze)

if True:  
    print("multi:",devices)
    model=nn.DataParallel(model,device_ids=devices)    

model=model.to(device)

if options.only_task and options.old_model is not None:
    print("fix para except task embedding")
    for name,para in model.named_parameters():
        if name.find("task_embed")==-1:
            para.requires_grad=False
        else:
            para.requires_grad=True
            print(name)
    
optimizer = optm.NoamOpt(options.d_model, options.factor, 4000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))