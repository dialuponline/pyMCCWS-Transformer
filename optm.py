import torch
import torch.optim as optim

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(sel