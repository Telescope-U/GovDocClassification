import pandas as pd
import jieba
import re
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchtext
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

train_df = pd.read_csv('../dataset/train.csv')
val_df = pd.read_csv('../dataset/val.csv')

# 将labels处理成数字组合
labels_series = train_df['labels'].str.split(',')
label_set = set()
for labels in labels_series:
    for label in labels:
        label_set.add(label)
label2idx = {label: i+1 for i, label in enumerate(label_set)}
label2idx[''] = 0
idx2label = {label2idx[label]: label for label in label2idx}

# 将label处理成数字化
def trans_label2idx(str_list):
    return [label2idx[label] for label in str_list]
train_df['labels'] = train_df['labels'].str.split(',').apply(trans_label2idx)
val_df['labels'] = val_df['labels'].str.split(',').apply(trans_label2idx)


""" 中文分词 """
def split_cn(text, sep=' '):
    punc = "[’!'\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+，。！？“”《》：、． \n（）〕〔. "
    a = [word for word in jieba.lcut(text) if (word not in punc) and (re.search('[0-9]|x', word) == None)]
    return sep.join(a)

train_df['article'] = train_df['article'].apply(lambda x: split_cn(x))
val_df['article'] = val_df['article'].apply(lambda x: split_cn(x))


""" 创建词表"""
def yield_tokens():
    for data in train_df['article']:
        tokens = [word for word in data.split()]
        yield tokens


vocabulary = build_vocab_from_iterator(yield_tokens(), min_freq=2, specials=["<unk>", "<pad>"])
vocabulary.set_default_index(vocabulary['<unk>'])


class TextDataset(Dataset):
    def __init__(self, data, idx2label=idx2label, vocab=vocabulary,  length=5000):
        self.data = data
        self.vocab = vocab
        self.idx2label = idx2label
        self.length = length

    def __getitem__(self, index):
        text = self.data.iloc[index]['article']
        words = text.split()
        words = self.tag(words)
        word_idxs = self.vocab.forward(words)
        labels = self.data.iloc[index]['labels']
        if len(labels) < 3:
            labels = labels + [0]

        return torch.tensor(word_idxs), F.one_hot(torch.tensor(labels), num_classes=len(idx2label)).sum(axis=0).float()

    def __len__(self):
        return self.data.shape[0]

    def tag(self, words):
        words = words[:self.length] + ["<pad>"] * (self.length - min(len(words), self.length))
        return words

train_dataset = TextDataset(train_df)
val_dataset = TextDataset(val_df)
