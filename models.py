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
            self.embed_dim+=word_embedding.shape[1]
        else:
            if bigram_embedding is not None:
                self.embed_dim+=bi_embedding.shape[1]
            else: self.embed_dim=d_model
            assert word_size is not None
            self.uni_embed = nn.Embedding(word_size,self.embed_dim)
            
        if bi_embedding is not None:    
            self.bi_embed = nn.Embedding.from_pretrained(torch.FloatTensor(bi_embedding), freeze=freeze)
            self.embed_dim += bi_embedding.shape[1]*2
            
        print("Trans Freeze",freeze,self.embed_dim)
        
        if d_model!=self.embed_dim:
            self.F=nn.Linear(self.embed_dim,d_model)
        else :
            self.F=None
            
        self.d_model = d_model

    def forward(self, task, uni, bi1=None, bi2=None):
        #print(task,uni.size(),bi1.size(),bi2.size())
        #print(bi1,bi2)
        #assert False
        y_task=self.task_embed(task[:,0:1])
        y=self.uni_embed(uni[:,1:])
        if bi1 is not None:
            assert self.bi_embed is not None
            
            y=torch.cat([y,self.bi_embed(bi1),self.bi_embed(bi2)],dim=-1)
            #y2=self.bi_embed(bi)
            #y=torch.cat([y,y2[:,:-1,:],y2[:,1:,:]],dim=-1)
            
        #y=torch.cat([y_task,y],dim=1)
        if self.F is not None:
            y=self.F(y)
        y=tor