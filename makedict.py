import utils
import pickle
import os
from utils import is_dataset_tag, make_sure_path_exists

path="data/joint-sighan-simp/raw/train-all.txt"

out_path="dict.pkl"

dic={}
tokens={}
with open(path, "r", encoding="utf-16") as f:
    for line in f.readlines():
        cur=line.strip().split(" ")
        name=cur[0]