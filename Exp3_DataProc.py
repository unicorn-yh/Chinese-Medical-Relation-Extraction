"""
这个文件中可以添加数据预处理相关函数和类等
如词汇表生成，Word转ID（即词转下标）等
此文件为非必要部分，可以将该文件中的内容拆分进其他部分
"""

import os
import jieba
import json
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from Exp3_DataSet import trainset, testset
from Exp3_Config import Training_Config

stopwords = []
config = Training_Config()
relation_to_id, id_to_relation = {}, {}

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

def vectorize_data(sentence, vocab_list):
    vector = np.zeros(config.embedding_dimension, dtype=int)
    index = 0
    new_list = list(jieba.cut(sentence, cut_all=False))
    new_list = remove_stopwords(new_list)
    for w in new_list:
        if w in vocab_list and index < 100:
            vector[index] = vocab_list.index(w)
            index += 1
    return vector

def get_location(head, tail, sentence):
    loc_vector = []
    head_loc = sentence.find(head) 
    tail_loc = sentence.find(tail)
    head_vec = [head_loc, head_loc+len(head)]
    tail_vec = [tail_loc, tail_loc+len(tail)]
    loc_vector.append(head_vec)
    loc_vector.append(";")
    loc_vector.append(tail_vec)
    loc_vector.append(";")
    return loc_vector

def get_relation_id():
    with open('data/rel2id.json',encoding='utf-8') as json_file:
        data = json.load(json_file)
        relation_dict = data[1]
    return relation_dict

    

def preprocess(dataset=trainset, keyword="train"):
    print("数据预处理开始......")
    
    tmp_list, vocab_list = [], []

    '''Get Relation'''
    if not os.path.exists("data/relation.txt"):
        file = open('data/relation.txt','w',encoding='utf-8')
        for key in relation_to_id.keys():
            file.write(key+"\n")
        file.close()


    '''Get Vocab List'''
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


    '''Get Sentence Vector / Word2Vec'''
    train_vec = []
    if not os.path.exists('data/trainX.txt'):
        for i in range(len(trainset)):
            vector = vectorize_data(trainset[i]['text'], vocab_list)
            train_vec.append(vector)
        train_vec = np.array(train_vec)
        np.savetxt("data/trainX.txt",train_vec,fmt='%s')
    
    train_vec = np.loadtxt("data/trainX.txt")
    train_vec = np.array(train_vec)
    print("训练集词嵌入:",train_vec.shape)


    '''Get Head / Tail Location'''
    loc_vec = []
    if not os.path.exists('data/location_vector.txt'):
        for i in range(len(trainset)):
            location = get_location(trainset[i]['head'],
                                    trainset[i]['tail'], 
                                    trainset[i]['text'])
            location.append(relation_to_id[trainset[i]['relation']])
            loc_vec.append(location)
        loc_vec = np.array(loc_vec)
        np.savetxt("data/location_vector.txt",loc_vec,fmt='%s')
    
    with open("data/location_vector.txt","r") as file:
        for line in file:
            tmp_vec = []
            loc1, loc2, relation = line.replace("\n","").split(" ; ")
            loc1 = loc1.strip('][').split(', ')
            loc2 = loc2.strip('][').split(', ')
            loc1 = [int(i) for i in loc1]
            loc2 = [int(i) for i in loc2]
            relation = int(relation)
            tmp_vec.append(loc1)
            tmp_vec.append(loc2)
            tmp_vec.append(relation)
            loc_vec.append(tmp_vec)
    loc_vec = np.array(loc_vec)
    print("头尾位置标签:",loc_vec.shape)


    '''Preprocess'''
    if not os.path.exists('data/train_data.jsonl'):
        data = []
        keylist = ["h","t","relation","text"]
        detailkey = ["name","pos"]
        with open('data/train_data.txt', mode='w', encoding='utf-8') as f:
            json.dump([], f)
        for i in range(len(trainset)):
            tmp_dict = {}
            tmp_dict = {key: None for key in keylist}
            tmp_dict["h"] = {key: None for key in detailkey}
            tmp_dict["t"] = {key: None for key in detailkey}
            tmp_dict["h"]["name"] = trainset[i]['head']
            tmp_dict["h"]["pos"] = loc_vec[i][0]
            tmp_dict["t"]["name"] = trainset[i]['tail']
            tmp_dict["t"]["pos"] = loc_vec[i][1]
            tmp_dict["relation"] = trainset[i]['relation']
            tmp_dict["text"] = trainset[i]['text']
            data.append(tmp_dict)
        
        with open('data/train_data.jsonl', mode='w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    data = []
    with open('data/train_data.jsonl', 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    

    

    

    '''inp = torch.tensor(train_vec[0]).long()
    embedding = nn.Embedding(config.vocab_size,config.embedding_dimension)'''
    


    print("数据预处理完毕！")



get_stopwords()
relation_to_id = get_relation_id()
preprocess()
