
import os
import sys
import errno
import time
import codecs
import numpy as np

def bmes_to_words(chars, tags):
    result = []
    if len(chars) == 0:
        return result
    word = chars[0]

    for c, t in zip(chars[1:], tags[1:]):
        if t.upper() == 'B' or t.upper() == 'S':
            result.append(word)
            word = ''
        word += c
    if len(word) != 0:
        result.append(word)

    return result

def bmes_to_index(tags):
    result = []
    if len(tags) == 0:
        return result
    word = (0, 0)

    for i, t in enumerate(tags):
        if i == 0:
            word = (0, 0)
        elif t.upper() == 'B' or t.upper() == 'S':
            result.append(word)
            word = (i, 0)
        word = (word[0], word[1] + 1)
    if word[1] != 0:
        result.append(word)
    return result

class CWSEvaluator:
    def __init__(self, i2t):
        self.correct_preds = 0.
        self.total_preds = 0.
        self.total_correct = 0.
        self.i2t = i2t

    def add_instance(self, pred_tags, gold_tags):
        pred_tags = [self.i2t[i] for i in pred_tags]
        gold_tags = [self.i2t[i] for i in gold_tags]
        # Evaluate PRF
        lab_gold_chunks = set(bmes_to_index(gold_tags))
        lab_pred_chunks = set(bmes_to_index(pred_tags))
        self.correct_preds += len(lab_gold_chunks & lab_pred_chunks)
        self.total_preds += len(lab_pred_chunks)
        self.total_correct += len(lab_gold_chunks)

    def result(self, percentage=True):
        p = self.correct_preds / self.total_preds if self.correct_preds > 0 else 0
        r = self.correct_preds / self.total_correct if self.correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        if percentage:
            p *= 100
            r *= 100
            f1 *= 100
        return p, r, f1

class CWS_OOV:
    def __init__(self,dic):
        self.dic=dic
        self.recall=0
        self.tot=0
        
    def update(self,gold_sent,pred_sent):
        i=0
        j=0
        id=0
        for w in gold_sent:
            if w not in self.dic:
                self.tot+=1
                while i+len(pred_sent[id])<=j:
                    i+=len(pred_sent[id])
                    id+=1
                if i==j and len(pred_sent[id])==len(w) and w.find(pred_sent[id])!=-1:
                    self.recall+=1                    
            j+=len(w)
        #print(gold_sent,pred_sent,self.tot)
        
    def oov(self, percentage=True):
        ins=1.0*self.recall/self.tot
        if percentage:
            ins *= 100