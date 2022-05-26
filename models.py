import fastNLP
import torch
import math
from fastNLP.modules.encoder.transformer import TransformerEncoder
from fastNLP.modules.decoder.crf import ConditionalRandomField
from fastNLP import Const
import copy
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import transformer

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class Embedding(nn.Module):
    def __init__(self,task_size, d_model, word_embedding=None, bi_embedding=None, word_size=None, freeze=True):
        super(Embedding, self).__init__()
        self.task_size=task_size        
        self.embed_dim = 0        
        
        self.task_embed = nn.Embedding(task_size,d_model)
        """
        if freeze:
            self.task_embed.weight.requires_grad = False
        """
        if word_embedding is not None:
            self.uni_embed = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding), freeze=freeze)
            self.embed_dim+=word_embedding.shap