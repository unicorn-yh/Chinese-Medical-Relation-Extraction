"""
这个文件中可以添加数据预处理相关函数和类等
如词汇表生成，Word转ID（即词转下标）等
此文件为非必要部分，可以将该文件中的内容拆分进其他部分
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from Exp3_DataSet import trainset, valset, testset
from Exp3_Config import Training_Config

config = Training_Config()

def get_location(head, tail, sentence):
    loc_vector = []
    head_loc = sentence.find(head) 
    tail_loc = sentence.find(tail)
    head_vec = [head_loc, head_loc+len(head)]
    tail_vec = [tail_loc, tail_loc+len(tail)]
    loc_vector.append(head_vec)
    loc_vector.append(";")
    loc_vector.append(tail_vec)
    return loc_vector


def relation2id():
    with open('data/rel2id.json',encoding='utf-8') as json_file:
        data = json.load(json_file)
        relation_dict = data[1]
    return relation_dict


def head_tail_location(dataset):
    '''Get Head / Tail Location'''
    loc_vec = []
    if dataset == trainset:
        key = "trainset"
    elif dataset == valset:
        key = "valset"
    elif dataset == testset:
        key = "testset"
    filename = 'data/'+key+'_location.txt'

    if not os.path.exists(filename):
        for i in range(len(dataset)):
            location = get_location(dataset[i]['head'],
                                    dataset[i]['tail'], 
                                    dataset[i]['text'])
            loc_vec.append(location)
        loc_vec = np.array(loc_vec)
        np.savetxt(filename,loc_vec,fmt='%s')
        return loc_vec
    
    with open(filename,"r") as file:
        for line in file:
            tmp_vec = []
            loc1, loc2 = line.replace("\n","").split(" ; ")
            loc1 = loc1.strip('][').split(', ')
            loc2 = loc2.strip('][').split(', ')
            loc1 = [int(i) for i in loc1]
            loc2 = [int(i) for i in loc2]
            tmp_vec.append(loc1)
            tmp_vec.append(loc2)
            loc_vec.append(tmp_vec)
    loc_vec = np.array(loc_vec, dtype=int)
    return loc_vec
    

class MyTokenizer(object):
    def __init__(self, pretrained_model_path=None, mask_entity=False):
        self.pretrained_model_path = pretrained_model_path or 'bert-base-chinese'
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
        self.mask_entity = mask_entity

    def tokenize(self, sentence, loc_vec):
        pos_head = loc_vec[0]
        pos_tail = loc_vec[1]

        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            pos_min = pos_head
            pos_max = pos_tail
            rev = False

        sent0 = self.bert_tokenizer.tokenize(sentence[:pos_min[0]])
        ent0 = self.bert_tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
        sent1 = self.bert_tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
        ent1 = self.bert_tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
        sent2 = self.bert_tokenizer.tokenize(sentence[pos_max[1]:])

        if rev:
            if self.mask_entity:
                ent0 = ['[unused6]']
                ent1 = ['[unused5]']
            pos_tail = [len(sent0), len(sent0) + len(ent0)]
            pos_head = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        else:
            if self.mask_entity:
                ent0 = ['[unused5]']
                ent1 = ['[unused6]']
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        tokens = sent0 + ent0 + sent1 + ent1 + sent2

        re_tokens = ['[CLS]']
        cur_pos = 0
        pos1 = [0, 0]
        pos2 = [0, 0]
        for token in tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                pos1[0] = len(re_tokens)
                re_tokens.append('[unused1]')
            if cur_pos == pos_tail[0]:
                pos2[0] = len(re_tokens)
                re_tokens.append('[unused2]')
            re_tokens.append(token)
            if cur_pos == pos_head[1] - 1:
                re_tokens.append('[unused3]')
                pos1[1] = len(re_tokens)
            if cur_pos == pos_tail[1] - 1:
                re_tokens.append('[unused4]')
                pos2[1] = len(re_tokens)
            cur_pos += 1
        re_tokens.append('[SEP]')
        return re_tokens[1:-1], pos1, pos2
    
    
def convert_pos_to_mask(e_pos, max_len=config.embedding_dimension):
    e_pos_mask = [0] * max_len
    for i in range(e_pos[0], e_pos[1]):
        e_pos_mask[i] = 1
    return e_pos_mask


def read_data(dataset, tokenizer=None, max_len=config.embedding_dimension):
    tokens_list = []
    e1_mask_list = []
    e2_mask_list = []
    tags = []
    loc_vec = head_tail_location(dataset)
    if tokenizer is None:
        tokenizer = MyTokenizer()
    for i in range(len(dataset)):
        tokens, pos_e1, pos_e2 = tokenizer.tokenize(dataset[i]['text'],loc_vec[i])
        if pos_e1[0] < max_len - 1 and pos_e1[1] < max_len and \
                pos_e2[0] < max_len - 1 and pos_e2[1] < max_len:
            tokens_list.append(tokens)
            e1_mask = convert_pos_to_mask(pos_e1, max_len)
            e2_mask = convert_pos_to_mask(pos_e2, max_len)
            e1_mask_list.append(e1_mask)
            e2_mask_list.append(e2_mask)
            if not dataset == testset:
                tag = dataset[i]['relation']
                tags.append(tag)
    return tokens_list, e1_mask_list, e2_mask_list, tags
    

class SentenceREDataset(Dataset):
    def __init__(self, dataset, pretrained_model_path=None, max_len=config.embedding_dimension):
        self.pretrained_model_path = pretrained_model_path or 'bert-base-chinese'
        self.tokenizer = MyTokenizer(pretrained_model_path=self.pretrained_model_path)
        self.max_len = max_len
        self.tokens_list, self.e1_mask_list, self.e2_mask_list, self.tags = read_data(dataset, tokenizer=self.tokenizer, max_len=self.max_len)
        self.tag2idx = relation2id()
        self.dataset = dataset

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_tokens = self.tokens_list[idx]
        sample_e1_mask = self.e1_mask_list[idx]
        sample_e2_mask = self.e2_mask_list[idx]
        encoded = self.tokenizer.bert_tokenizer.encode_plus(sample_tokens, max_length=self.max_len, pad_to_max_length=True)
        sample_token_ids = encoded['input_ids']
        sample_token_type_ids = encoded['token_type_ids']
        sample_attention_mask = encoded['attention_mask']

        if self.dataset == testset:
            sample = {
                'token_ids': torch.tensor(sample_token_ids),
                'token_type_ids': torch.tensor(sample_token_type_ids),
                'attention_mask': torch.tensor(sample_attention_mask),
                'e1_mask': torch.tensor(sample_e1_mask),
                'e2_mask': torch.tensor(sample_e2_mask)
            }
        else:
            sample_tag = self.tags[idx]
            sample_tag_id = self.tag2idx[sample_tag]
            sample = {
                'token_ids': torch.tensor(sample_token_ids),
                'token_type_ids': torch.tensor(sample_token_type_ids),
                'attention_mask': torch.tensor(sample_attention_mask),
                'e1_mask': torch.tensor(sample_e1_mask),
                'e2_mask': torch.tensor(sample_e2_mask),
                'tag_id': torch.tensor(sample_tag_id)
            }
        
        return sample


print("数据预处理开始......")
print("预处理前的训练集、验证集、测试集大小:",len(trainset),",",len(valset),",",len(testset)) 
train_data = SentenceREDataset(trainset)
val_data = SentenceREDataset(valset)
test_data = SentenceREDataset(testset)
print("预处理后的训练集、验证集、测试集大小:",len(train_data),",",len(val_data),",",len(test_data)) 
print("数据预处理完毕！")


