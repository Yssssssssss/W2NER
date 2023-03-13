import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
from gensim.models import KeyedVectors
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import utils
import requests

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype='int64')  # 这里的限制是句子长度不能大于1000
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]


def collate_fn(data):
    bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text = map(list, zip(*data))

    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text):
        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.entity_text = entity_text

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.grid_labels[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               torch.LongTensor(self.dist_inputs[item]), \
               self.sent_length[item], \
               self.entity_text[item]

    def __len__(self):
        return len(self.bert_inputs)


def process_bert(data, tokenizer, vocab):
    bert_inputs = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []

    for index, instance in enumerate(tqdm(data)):  # 主要数据处理在这部分
        if len(instance['sentence']) == 0:
            continue
        tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])

        length = len(instance['sentence'])
        _grid_labels = np.zeros((length, length), dtype=np.int)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
        _dist_inputs = np.zeros((length, length), dtype=np.int)
        _grid_mask2d = np.ones((length, length), dtype=np.bool)

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19

        for entity in instance["ner"]:
            index = entity["index"]
            for i in range(len(index)):
                if i + 1 >= len(index):
                    break
                _grid_labels[index[i], index[i + 1]] = 1
            _grid_labels[index[-1], index[0]] = vocab.label_to_id(entity["type"])

        _entity_text = set([utils.convert_index_to_text(e["index"], vocab.label_to_id(e["type"]))
                            for e in instance["ner"]])

        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)
        entity_text.append(_entity_text)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


def fill_vocab(vocab, dataset):
    entity_num = 0
    for instance in dataset:
        for entity in instance["ner"]:
            vocab.add_label(entity["type"])
        entity_num += len(instance["ner"])
    # vocab.add_label('ud')#尝试在label_num上补全 UD  必须补在  fill_vocab之中，因为fill_vocab会生成id2label和label2id
    return entity_num


def load_data_bert(config):
    with open('./data/{}/train.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/{}/dev.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open('./data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")

    vocab = Vocabulary()
    train_ent_num = fill_vocab(vocab, train_data)
    dev_ent_num = fill_vocab(vocab, dev_data)
    test_ent_num = fill_vocab(vocab, test_data)

    table = pt.PrettyTable([config.dataset, 'sentences', 'entities'])
    table.add_row(['train', len(train_data), train_ent_num])
    table.add_row(['dev', len(dev_data), dev_ent_num])
    table.add_row(['test', len(test_data), test_ent_num])
    config.logger.info("\n{}".format(table))
    config.label_num = len(vocab.label2id)
    config.vocab = vocab

    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))
    return (train_dataset, dev_dataset, test_dataset), (train_dataset, dev_data, test_dataset)


def load_data_bert_predict(config):
    with open('./data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")
    vocab = Vocabulary()
    test_ent_num = fill_vocab(vocab, test_data)
    # 100 舍弃长句子 保留十分之一
    # vocab.id2label={0: '<pad>', 1: '<suc>', 2: 'cn', 3: 'sd', 4: 'ts', 5: 'eh', 6: 'lhs', 7: 'ed', 8: 'ap', 9: 'ps', 10: 'pr', 11: 'ths', 12: 'thr', 13: 'pe', 14: 'htp', 15: 'ra', 16: 'cd', 17: 'rs', 18: 'ltp', 19: 'rd', 20: 'tps', 21: 'li', 22: 'fs', 23: 'ud'}
    # vocab.label2id={'<pad>': 0, '<suc>': 1, 'cn': 2, 'sd': 3, 'ts': 4, 'eh': 5, 'lhs': 6, 'ed': 7, 'ap': 8, 'ps': 9, 'pr': 10, 'ths': 11, 'thr': 12, 'pe': 13, 'htp': 14, 'ra': 15, 'cd': 16, 'rs': 17, 'ltp': 18, 'rd': 19, 'tps': 20, 'li': 21, 'fs': 22, 'ud': 23}
    # 100 舍弃长句子
    # vocab.id2label = {0: '<pad>', 1: '<suc>', 2: 'cn', 3: 'sd', 4: 'eh', 5: 'ts', 6: 'lhs', 7: 'ed', 8: 'ap', 9: 'li',
    #                   10: 'ps',
    #                   11: 'ths', 12: 'thr', 13: 'pr', 14: 'pe', 15: 'cd', 16: 'htp', 17: 'ltp', 18: 'rs', 19: 'ra',
    #                   20: 'tps', 21: 'rd',
    #                   22: 'fs', 23: 'ud'}
    # vocab.label2id = {'<pad>': 0, '<suc>': 1, 'cn': 2, 'sd': 3, 'eh': 4, 'ts': 5, 'lhs': 6, 'ed': 7, 'ap': 8, 'li': 9,
    #                   'ps': 10, 'ths': 11, 'thr': 12, 'pr': 13, 'pe': 14, 'cd': 15, 'htp': 16, 'ltp': 17, 'rs': 18,
    #                   'ra': 19, 'tps': 20, 'rd': 21, 'fs': 22, 'ud': 23}
    # 100 处理长句 保留五分之一全O标签
    vocab.id2label = {0: '<pad>', 1: '<suc>', 2: 'cn', 3: 'sd', 4: 'eh', 5: 'ts', 6: 'ed', 7: 'ap', 8: 'lhs', 9: 'li',
                      10: 'fs',
                      11: 'ps', 12: 'thr', 13: 'ths', 14: 'ud', 15: 'pr', 16: 'pe', 17: 'tps', 18: 'cd', 19: 'rs',
                      20: 'htp', 21: 'ltp',
                      22: 'ra', 23: 'rd'}
    vocab.label2id = {'<pad>': 0, '<suc>': 1, 'cn': 2, 'sd': 3, 'eh': 4, 'ts': 5, 'ed': 6, 'ap': 7, 'lhs': 8, 'li': 9,
                      'fs': 10,
                      'ps': 11, 'thr': 12, 'ths': 13, 'ud': 14, 'pr': 15, 'pe': 16, 'tps': 17, 'cd': 18, 'rs': 19,
                      'htp': 20, 'ltp': 21,
                      'ra': 22, 'rd': 23}
    table = pt.PrettyTable([config.dataset, 'sentences', 'entities'])
    table.add_row(['test', len(test_data), test_ent_num])
    config.logger.info("\n{}".format(table))
    config.label_num = len(vocab.label2id)  # 预测的时候维度对不上是因为label_num不全  缺少UD标签  W2NER的label中自带的有两个 <pad> 和<suc>
    config.vocab = vocab
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))
    return (test_dataset,), (test_data,)  # 单个元素保存元组形式，需要加一个 ,
