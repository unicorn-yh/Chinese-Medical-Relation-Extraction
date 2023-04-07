"""
这个文件中可以添加数据预处理相关函数和类等
如词汇表生成，Word转ID（即词转下标）等
此文件为非必要部分，可以将该文件中的内容拆分进其他部分
"""

import os
import jieba
import numpy as np
from Exp3_DataSet import trainset, testset
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
    vector = []
    new_list = list(jieba.cut(head, cut_all=False)) + list(jieba.cut(tail, cut_all=False))
    new_list = remove_stopwords(new_list)
    

if True:
    print("数据预处理开始......")
    get_stopwords()
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

    #print(trainset[0])
    train_vec = []
    if not os.path.exists('data/trainX.txt'):
        for i in range(len(trainset)):
            vector = vectorize_data(trainset[i]['text'], vocab_list)
            train_vec.append(vector)
        train_vec = np.array(train_vec)
        np.savetxt("data/trainX.txt",train_vec,fmt='%s')
    
    train_vec = np.loadtxt("data/trainX.txt")
    train_vec = np.array(train_vec)
    print(train_vec.shape)

    print("数据预处理完毕！")
