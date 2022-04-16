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
i