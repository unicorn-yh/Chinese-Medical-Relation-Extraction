"""
这个文件中可以添加数据预处理相关函数和类等
如词汇表生成，Word转ID（即词转下标）等
此文件为非必要部分，可以将该文件中的内容拆分进其他部分
"""

import re
import jieba
from Exp3_DataSet import trainset, testset

stopwords = []

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
        

'''利用语义角色标注,直接获取主谓宾三元组,基于A0,A1,A2'''
def ruler1(self, words, postags, roles_dict, role_index):
    v = words[role_index]
    role_info = roles_dict[role_index]
    if 'A0' in role_info.keys() and 'A1' in role_info.keys():
        s = ''.join([words[word_index] for word_index in range(role_info['A0'][1], role_info['A0'][2] + 1) if
                        postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
        o = ''.join([words[word_index] for word_index in range(role_info['A1'][1], role_info['A1'][2] + 1) if
                        postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
        if s and o:
            return '1', [s, v, o]
    return '4', []

if __name__ == '__main__':
    print("数据预处理开始......")
    get_stopwords()
    new = list(jieba.cut(trainset[0]['text'], cut_all=False))
    #print(new)
    print(remove_stopwords(new))
    print("数据预处理完毕！")
