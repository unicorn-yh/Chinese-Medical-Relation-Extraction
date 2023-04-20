"""
这个文件中可以添加数据预处理相关函数和类等
如词汇表生成，Word转ID（即词转下标）等
此文件为非必要部分，可以将该文件中的内容拆分进其他部分
"""

import os
import json
import numpy as np
import torch
import jieba
from torch.utils.data import Dataset
from transformers import BertTokenizer
from Exp3_DataSet import trainset, valset, testset
from Exp3_Config import Training_Config

stopwords = []
config = Training_Config()


def get_stopwords():
    with open("data/stopwords_utf8.txt","r",encoding="utf-8") as file:
        for char in file:
            stopwords.append(char.replace("\n",""))


def remove_stopwords(sent_list):
    new_list = []
    for char in sent_list:
        isstopword = False
        for ch in stopwords:
            if char == ch:
                isstopword = True
                continue
        if not isstopword:
            new_list.append(char)
    return new_list


def add_to_vocab(vocab_list, new_list):
    for word in new_list:
        if not word in vocab_list:
            if len(vocab_list) < config.vocab_size:
                vocab_list.append(word)
    return vocab_list


def get_vocab_list():
    '''Get Vocab List'''
    tmp_list, vocab_list = [], []
    if not os.path.exists('data/vocab_list.txt'):
        for i in range(len(trainset)):
            if len(tmp_list) > config.vocab_size:
                break
            new_list = list(jieba.cut(trainset[i]['text'], cut_all=False))
            new_list = remove_stopwords(new_list)
            tmp_list = add_to_vocab(tmp_list, new_list)

        file = open('data/vocab_list.txt','w',encoding='utf-8')
        for word in tmp_list:
            file.write(word+"\n")
        file.close()

    with open('data/vocab_list.txt','r', encoding='utf-8') as file:
        for line in file:
            vocab_list.append(line.replace("\n",""))

    print("词汇表:",len(vocab_list))
    file.close()
    return vocab_list


def relation2id():
    with open('data/rel2id.json',encoding='utf-8') as json_file:
        data = json.load(json_file)
        relation_dict = data[1]
    return relation_dict


def id2relation():
    with open('data/rel2id.json',encoding='utf-8') as json_file:
        data = json.load(json_file)
        relation_dict = data[0]
    return relation_dict


def word2vec():
    word2vec_dict = {}
    with open('data/skip-gram-model.txt','r',encoding='utf-8') as file:
        for line in file:
            tmp = line.index("[")
            word2vec_dict[line[0:tmp]] = line[tmp+1:len(line)-2]
    return word2vec_dict


def pre_embedding(word2vec_dict):
    vocab_list = []
    with open('data/vocab_list.txt','r',encoding='utf-8') as file:
        for line in file:
            vocab_list.append(line.strip("\n"))
    word2index = {word:index for index,word in enumerate(vocab_list)}
    index2word = {index:word for index,word in enumerate(vocab_list)}
    word2index["BLANK"] = len(word2index) + 1
    word2index["UNKNOWN"] = len(word2index) + 1
    index2word[len(index2word) + 1] = ["BLANK"]
    index2word[len(index2word) + 1] = ["UNKNOWN"]

    unknown_pre, pre_embed = [],[]
    unknown_pre.extend([1]*config.embedding_dimension)
    pre_embed.append(unknown_pre)
    for word in word2index.keys():
        if word in word2vec_dict.keys():
            pre_embed.append(torch.FloatTensor(eval(word2vec_dict[word])))
        else:
            pre_embed.append(torch.FloatTensor(unknown_pre))
    pre_embed = torch.FloatTensor(pre_embed)

    return pre_embed, word2index, index2word


def get_location(head, tail, sentence):
    loc_vector = []
    head_loc = sentence.find(head) 
    tail_loc = sentence.find(tail)
    out_max = config.max_sentence_length + 1
    if head_loc == -1:
        head_vec = [-1,-1]
    else:
        head_vec = [head_loc, head_loc+len(head)]
    if tail_loc == -1:
        tail_vec = [out_max, out_max]
    else:   
        tail_vec = [tail_loc, tail_loc+len(tail)]
    loc_vector.append(head_vec)
    loc_vector.append(";")
    loc_vector.append(tail_vec)
    return loc_vector


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


def get_tensordata(dataset,loc_vec,word2index,relation2id_dict):
    '''Get data in Tensor format'''
    tensor_data = []
    sent_len = config.max_sentence_length

    for i in range(len(dataset)):
        segmented = list(jieba.cut(dataset[i]['text'], cut_all=False))
        segmented = remove_stopwords(segmented)
        pos1,pos2,sent,label = [],[],[],[]
        k = 1
        for word in segmented:
            if word in word2index.keys():
                sent.append(word2index[word])
            else:
                sent.append(word2index["UNKNOWN"])
            pos1.append(pos_feature(k-loc_vec[i][0][0]+1))
            pos2.append(pos_feature(k-loc_vec[i][1][0]+1))
            k += 1
        if len(segmented) < sent_len:
            sent.extend([word2index["BLANK"]]*(sent_len-len(sent)))
            pos1.extend([101]*(sent_len-len(pos1)))
            pos2.extend([101]*(sent_len-len(pos2)))

        sent_data = np.zeros(sent_len)
        pos1_data = np.zeros(sent_len)
        pos2_data = np.zeros(sent_len)
        sent_data[:sent_len] = sent[:sent_len]
        pos1_data[:sent_len] = pos1[:sent_len]
        pos2_data[:sent_len] = pos2[:sent_len]
        sent_data = torch.LongTensor(sent_data)
        pos1_data = torch.LongTensor(pos1_data)
        pos2_data = torch.LongTensor(pos2_data)
        tmp_dict = {}
        tmp_dict = {'text':sent_data,'pos1':pos1_data,'pos2':pos2_data}

        if not dataset == testset:
            label.append(relation2id_dict[dataset[i]['relation']])
            label_data = torch.LongTensor(label)
            tensor_data.append((tmp_dict,label_data))
        else:
            tensor_data.append(tmp_dict)
        
    return tensor_data


def pos_feature(x):
    if x < -50:
        return 0
    elif x >= -50 and x <= 50:
        return x + 50
    else:
        return 50 * 2  


def get_data(dataset,word2index,relation2id_dict):
    loc_vector = head_tail_location(dataset)
    tensordata = get_tensordata(dataset,loc_vector,word2index,relation2id_dict)
    return tensordata


print("数据预处理开始......")
print("预处理前的训练集、验证集、测试集大小:",len(trainset),",",len(valset),",",len(testset))
get_stopwords()
get_vocab_list()
word2vec_dict = word2vec()
relation2id_dict = relation2id()
pre_embed, word2index, index2word = pre_embedding(word2vec_dict)
train_data = get_data(trainset,word2index,relation2id_dict)
test_data = get_data(testset,word2index,relation2id_dict)
val_data = get_data(valset,word2index,relation2id_dict)
print("预处理后的训练集、验证集、测试集大小:",len(train_data),",",len(val_data),",",len(test_data)) 
print("数据预处理完毕！")









