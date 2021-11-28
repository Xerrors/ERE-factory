"""
Dataset for rel extraction


"""

import os
import json
from typing import DefaultDict
import torch
from torch.utils import data
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer

import numpy as np
from config import Label2IdxSub, Label2IdxObj

class Features(object):

    def __init__(self, text, tokens, ids, mask, 
        seq_tag=None, corres_tag=None, table_tag=None, rel=None,
        triples=None, rel_tag=None, ent_tag=None, neg_mask=None,
    ):
        self.text = text
        self.tokens = tokens
        self.ids = ids
        self.mask = mask
        self.seq_tag = seq_tag
        self.corres_tag = corres_tag
        self.table_tag = table_tag
        self.rel = rel
        self.triples = triples
        self.rel_tag = rel_tag
        self.ent_tag = ent_tag
        self.neg_mask = neg_mask



class CustomDataset(Dataset):
    """ Custom dataset for various programs"""

    def __init__(self, config, data_sign='train'):
        self.data_sign = data_sign
        self.path = os.path.join(config.data_dir, '{}_triples.json'.format(data_sign))
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_dir, do_lower_case=False, do_basic_tokenize=False)

        print("Loading data for {} from {}...".format(self.data_sign, self.path))
        with open(os.path.join(config.data_dir, 'rel2id.json'), mode='r', encoding='utf-8') as f:
            self.rel2idx = json.load(f)[-1]
            self.idx2rel = dict(zip(self.rel2idx.values(), self.rel2idx.keys()))
            self.rel_num = len(self.rel2idx)

        with open(self.path, 'r', encoding='utf-8') as f:
            self.ori_data = json.load(f)

        self.data = []
        for sent in self.ori_data:
            self.data.extend(self.tokenize(sent, config))

    def tokenize(self, sent, config):
        """ convert sents to multi features, need to rewrite"""
        items = []
        return items

    def collate_fn_test(self, data):
        batch_max_len = max([len(d.tokens) for d in data])
        ids = torch.from_numpy(seq_padding(batch_max_len, [d.ids for d in data])).long().cuda()
        masks = torch.from_numpy(seq_padding(batch_max_len, [d.mask for d in data])).long().cuda()
        triples = [d.triples for d in data]
        tokens = [d.tokens for d in data]
        texts = [d.text for d in data]
        coll_data = [ids, masks, triples, tokens, texts]
        return coll_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]



class SentDataset(CustomDataset):
    """ Custom sentense dataset that each item contains all triples of one sentense"""

    def __init__(self, config, data_sign='train') -> None:
        super().__init__(config, data_sign)

    def tokenize(self, sent, config):
        text = sent['text']
        
        tokens = ['[CLS]'] + self.tokenizer.tokenize(text) + ['[SEP]']
        # triples = sent['triple_list']
        triples = [convert_triple_to_idx(tokens, self.tokenizer, triple, self.rel2idx) for triple in sent['triple_list']]

        len_token = len(tokens)
        if len_token > config.max_text_len:
            tokens = tokens[:config.max_text_len]

        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        mask = [1] * len(tokens)

        if self.data_sign != 'train':
            return [Features(text, tokens, ids, mask, triples=triples)]

        # 1. table_loss & neg_mask
        table_tag = np.zeros([len_token, len_token, self.rel_num])
        neg_mask = np.zeros([len_token, len_token, self.rel_num])
        # 2. corres_tag loss
        corres_tag = np.zeros([len_token, len_token])
        # 3. entity loss
        ent_tag = [np.zeros([len_token, self.rel_num]), np.zeros([len_token, self.rel_num])] # [subs, objs]

        for rel_id, sub_head, sub_tail, obj_head, obj_tail in triples:
            table_tag[sub_head:sub_tail, obj_head:obj_tail, rel_id] = 1
            ent_tag[0][sub_head:sub_tail, rel_id] = 1
            ent_tag[1][obj_head:obj_tail, rel_id] = 1
            corres_tag[sub_head:sub_tail, obj_head:obj_tail] = 1

            neg_mask[sub_head-1:sub_tail+1, obj_head-1:obj_tail+1, rel_id] = 1
            neg_mask[sub_head-1:sub_tail+1, obj_head-1:obj_tail+1, np.random.randint(self.rel_num)] = 1
            neg_mask[obj_head-1:obj_tail+1, sub_head-1:sub_tail+1, rel_id] = 1

        # NOTE: enable sym
        if config.args.use_symmetries:
            corres_tag += corres_tag.T * (0.5 if config.args.use_symmetries == 'symmetries_rate' else 1)


        rel_tag = [0] * self.rel_num
        for triple in triples:
            rel_tag[triple[0]] = 1

        return [Features(text, tokens, ids, mask, 
            table_tag=table_tag,
            corres_tag=corres_tag,
            triples=triples,
            rel_tag=rel_tag,
            ent_tag=ent_tag,
            neg_mask=neg_mask,
        )]

    def collate_fn_train(self, data):
        tokens_length = [len(d.tokens) for d in data]
        batch_max_len = max(tokens_length)
        ids = torch.from_numpy(seq_padding(batch_max_len, [d.ids for d in data])).long().cuda()
        masks = torch.from_numpy(seq_padding(batch_max_len, [d.mask for d in data])).long().cuda()
        ent_tags = torch.from_numpy(seq_padding(batch_max_len, [d.ent_tag for d in data], padding=[0]*self.rel_num, dim=2)).float().cuda()
        corres_tags = torch.from_numpy(seq_padding(batch_max_len, [d.corres_tag for d in data], dim=[1,2])).float().cuda()
        rel_tags = torch.tensor([d.rel_tag for d in data]).float().cuda()
        table_tags = torch.from_numpy(seq_padding(
            batch_max_len, [d.table_tag for d in data], dim="table_tag"
        )).float().cuda()
        neg_masks = torch.from_numpy(seq_padding(
            batch_max_len, [d.neg_mask for d in data], dim="table_tag"
        )).float().cuda()
        coll_data = [ids, masks, table_tags, neg_masks, corres_tags, rel_tags, ent_tags]
        return coll_data

class TripleDataset(CustomDataset):
    """ Custom triples dataset that each item only contains one triple of a sentense"""

    def __init__(self, config, data_sign='train'):
        super().__init__(config, data_sign=data_sign)

    def tokenize(self, sent, config):
        text = sent['text']
        
        tokens = self.tokenizer.tokenize(text)
        # triples = sent['triple_list']
        triples = [convert_triple_to_idx(tokens, self.tokenizer, triple, self.rel2idx) for triple in sent['triple_list']]

        len_token = len(tokens)
        if len_token > config.max_text_len:
            tokens = tokens[:config.max_text_len]

        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        mask = [1] * len(tokens)

        if self.data_sign != 'train':
            return [Features(text, tokens, ids, mask, triples)]

        items = [] # features for training

        corres_tag = np.zeros([len_token, len_token])
        for rel_id, sub_head, sub_tail, obj_head, obj_tail in triples:
            corres_tag[sub_head, obj_head] = 1

            if config.args.use_symmetries == 'symmetries':
                corres_tag[sub_head, obj_head] = 1
            elif config.args.use_symmetries == 'symmetries_rate':
                corres_tag[sub_head, obj_head] = 0.8

        rel_tag = [0] * self.rel_num
        for triple in triples:
            rel_tag[triple[0]] = 1

        for rel_id, sub_head, sub_tail, obj_head, obj_tail in triples:
            tags_sub = len_token * [Label2IdxSub['O']]
            tags_obj = len_token * [Label2IdxSub['O']]

            tags_sub[sub_head] = Label2IdxSub['B-H']
            tags_obj[obj_head] = Label2IdxObj['B-T']
            tags_sub[sub_head+1:sub_tail] = [Label2IdxSub['I-H']] * (sub_tail - sub_head - 1)
            tags_obj[obj_head+1:obj_tail] = [Label2IdxObj['I-T']] * (obj_tail - obj_head - 1)
            seq_tag = [tags_sub, tags_obj]
            
            items.append(Features(text, tokens, ids, mask, 
                seq_tag=seq_tag,
                corres_tag=corres_tag,
                rel=rel_id,
                triples=triples,
                rel_tag=rel_tag
            ))
        return items

    def collate_fn_train(self, data):
        tokens_length = [len(d.tokens) for d in data]
        batch_max_len = max(tokens_length)
        ids = torch.from_numpy(seq_padding(batch_max_len, [d.ids for d in data])).long().cuda()
        masks = torch.from_numpy(seq_padding(batch_max_len, [d.mask for d in data])).long().cuda()
        seq_tags = torch.from_numpy(seq_padding(batch_max_len, [d.seq_tag for d in data], dim=2)).long().cuda()
        rels = torch.tensor([d.rel for d in data]).long().cuda()
        corres_tags = torch.from_numpy(seq_padding(batch_max_len, [d.corres_tag for d in data], dim=[1,2])).float().cuda()
        rel_tags = torch.tensor([d.rel_tag for d in data]).float().cuda()
        coll_data = [ids, masks, seq_tags, rels, corres_tags, rel_tags]
        return coll_data


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

def convert_triple_to_idx(text, tokenizer, triple, rel2idx):
    sub, rel, obj = triple
    rel_id = rel2idx[rel]

    sub = tokenizer.tokenize(sub)
    obj = tokenizer.tokenize(obj)
    sub_head = find_head_idx(text, sub)
    obj_head = find_head_idx(text, obj)

    if sub_head == -1 or obj_head == -1:
        raise ValueError("can not find ({}, {}) in {}".format(sub, obj, text))

    sub_tail = sub_head + len(sub) # sub_tail 处并不是实体
    obj_tail = obj_head + len(obj) # obj_tail 处并不是实体

    return rel_id, sub_head, sub_tail, obj_head, obj_tail

def convert_idx_to_triple(tokens, idx, idx2rel):
    rel_id, sub_head, sub_tail, obj_head, obj_tail = idx
    rel = idx2rel[rel_id]
    sub = " ".join(tokens[sub_head:sub_tail]).replace(' ##', '')
    obj = " ".join(tokens[obj_head:obj_tail]).replace(' ##', '')
    return sub, rel, obj

def seq_padding(length, batchs, padding=0, dim=1):
    if dim == 1:
        arr = np.array([
            list(seq) + [padding for _ in range(length - len(seq))] for seq in batchs
        ])
        return arr
    elif dim == 2:
        arr = np.array([[
            list(seq) + [padding for _ in range(length - len(seq))] for seq in batch_dim2
        ] for batch_dim2 in batchs])
        return arr
    elif dim == [1, 2] and padding == 0:
        arr = []
        for mat in batchs:
            pad_width = (((0,int(length-len(mat))), (0,int(length-len(mat)))))
            arr.append(np.pad(mat, pad_width=pad_width, constant_values=(0,0)))
        return np.array(arr)
    
    elif dim == 'table_tag':
        arr = []
        for tab in batchs:
            l, _, rel = tab.shape
            tab = np.concatenate([tab, np.zeros([length-l, l, rel])], axis=0)
            tab = np.concatenate([tab, np.zeros([length, length-l, rel])], axis=1)
            arr.append(tab)
        return np.array(arr)
