
import os
import re
import opencc
from tqdm import tqdm
from utils import make_sure_path_exists, append_tags

t2s = opencc.OpenCC('t2s')
s2t = opencc.OpenCC('s2t')

def normalize(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


def preprocess(text):
    rNUM = u'(-|\+)?\d+((\.|·)\d+)?%?'
    rENG = u'[A-Za-z_.]+'
    sent = normalize(text.strip()).split()
    new_sent = []
    for word in sent:
        word = re.sub(u'\s+', '', word, flags=re.U)
        word = re.sub(rNUM, u'0', word, flags=re.U)
        word = re.sub(rENG, u'X', word)
        new_sent.append(word)
    return new_sent


def to_sentence_list(text, split_long_sentence=False):
    text = preprocess(text)
    delimiter = set()
    delimiter.update(u'。！？：；…、，（）”’,;!?、,')
    delimiter.add(u'……')
    sent_list = []
    sent = []
    for word in text:
        sent.append(word)
        if word in delimiter or (split_long_sentence and len(sent) >= 50):
            sent_list.append(sent)
            sent = []

    if len(sent) > 0:
        sent_list.append(sent)

    return sent_list


def convert_file(srcfile, desfile, split_long_sentence=False, encode='utf-8'):
    with open(srcfile, encoding=encode) as src, open(desfile, 'w',encoding="utf-16") as des:
        for line in src:
            for sent in to_sentence_list(line, split_long_sentence):
                des.write(' '.join(sent) + '\n')
                # if len(''.join(sent)) > 200:
                #     print(' '.join(sent))

def split_train_dev(dataset,encode='utf-16'):
    root = 'data/' + dataset + '/raw/'
    with open(root + 'train-all.txt',encoding=encode) as src, open(root + 'train.txt', 'w',encoding="utf-16") as train, open(root + 'dev.txt','w',encoding="utf-16") as dev:
        lines = src.readlines()
        idx = int(len(lines) * 0.9)
        for line in lines[: idx]:
            train.write(line)
        for line in lines[idx:]:
            dev.write(line)


def combine_files(one, two, out):
    if os.path.exists(out):
        os.remove(out)
    with open(one) as one, open(two) as two, open(out, 'a') as out:
        for line in one:
            out.write(line)
        for line in two:
            out.write(line)


def bmes_tag(input_file, output_file,encode="utf-16"):
    with open(input_file,encoding=encode) as input_data, open(output_file, 'w',encoding="utf-16") as output_data:
        for line in input_data:
            word_list = line.strip().split()
            for word in word_list:
                if len(word) == 1 or (len(word) > 2 and word[0] == '<' and word[-1] == '>'):
                    output_data.write(word + "\tS\n")
                else:
                    output_data.write(word[0] + "\tB\n")
                    for w in word[1:len(word) - 1]:
                        output_data.write(w + "\tM\n")
                    output_data.write(word[len(word) - 1] + "\tE\n")
            output_data.write("\n")


def make_bmes(dataset='pku',encode="utf-16"):
    path = 'data/' + dataset + '/'
    make_sure_path_exists(path + 'bmes')
    bmes_tag(path + 'raw/train.txt', path + 'bmes/train.txt',encode)
    bmes_tag(path + 'raw/train-all.txt', path + 'bmes/train-all.txt',encode)
    bmes_tag(path + 'raw/dev.txt', path + 'bmes/dev.txt',encode)
    bmes_tag(path + 'raw/test.txt', path + 'bmes/test.txt',encode)


def convert_sighan2005_dataset(dataset):
    root = 'data/' + dataset
    make_sure_path_exists(root)
    make_sure_path_exists(root + '/raw')
    convert_file('data/sighan2005/{}_training.utf8'.format(dataset), 'data/{}/raw/train-all.txt'.format(dataset), True)
    convert_file('data/sighan2005/{}_test_gold.utf8'.format(dataset), 'data/{}/raw/test.txt'.format(dataset), False)
    split_train_dev(dataset)


def convert_sighan2008_dataset(dataset, utf=16):
    root = 'data/' + dataset
    make_sure_path_exists(root)
    make_sure_path_exists(root + '/raw')
    convert_file('data/sighan2008/{}_seg_truth&resource/{}_train_utf{}.seg'.format(dataset, dataset, utf),
                 'data/{}/raw/train-all.txt'.format(dataset), True, 'utf-{}'.format(utf))
    convert_file('data/sighan2008/{}_seg_truth&resource/{}_truth_utf{}.seg'.format(dataset, dataset, utf),
                 'data/{}/raw/test.txt'.format(dataset), False, 'utf-{}'.format(utf))
    split_train_dev(dataset)


def convert_sxu():
    dataset = 'sxu'