'''用于获取数据集和生成 skip-gram 模型'''

import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import numpy as np
from Exp3_Config import Training_Config

config = Training_Config()
device = 'cuda' if config.cuda else 'cpu'


# 训练集和验证集
class TextDataSet(Dataset):
    def __init__(self, filepath):
        lines = open(filepath, 'r', encoding='utf-8').readlines()
        self.original_data = []
        for line in lines:
            tmp = {}
            line = line.split('\t')
            tmp['head'] = line[0]
            tmp['tail'] = line[1]
            tmp['relation'] = line[2]
            tmp['text'] = line[3][:-1]
            self.original_data.append(tmp)

    def __getitem__(self, index):
        return self.original_data[index]

    def __len__(self):
        return len(self.original_data)


# 测试集是没有标签的，因此函数会略有不同
class TestDataSet(Dataset):
    def __init__(self, filepath):
        lines = open(filepath, 'r', encoding='utf-8').readlines()
        self.original_data = []
        for line in lines:
            tmp = {}
            line = line.split('\t')
            tmp['head'] = line[0]
            tmp['tail'] = line[1]
            tmp['text'] = line[2][:-1]
            self.original_data.append(tmp)

    def __getitem__(self, index):
        return self.original_data[index]

    def __len__(self):
        return len(self.original_data)
    

trainset = TextDataSet(filepath="./data/data_train.txt")
valset = TextDataSet(filepath="./data/data_val.txt")
testset = TestDataSet(filepath="./data/test_exp3.txt")


'''--------------------------- 生成 Skip-gram 模型 ------------------------------'''
C = 3 
num_sampled = 64  # 负采样个数   
BATCH_SIZE = 1024  
EMBEDDING_SIZE = 100  #想要的词向量长度

def get_all_words(dataset,all_words_list):   # 获取所有词
    for i in range(len(dataset)):
        seg_list = list(jieba.cut(dataset[i]['text'], cut_all=False))
        all_words_list += seg_list
    return all_words_list


def build_dataset(all_words_list,word_num):
    word_count = [["UNKNOWN",-1]]    # 记录词频
    word_count.extend(Counter(all_words_list).most_common(word_num-1))
    word_freq = {}   # 根据词频记录词的排名
    word_data = []   # 根据词表记录词频
    unknown_count = 0
    for word, count in word_count:
        word_freq[word] = len(word_freq)
    for word in all_words_list:
        if word in word_freq.keys():
            word_data.append(word_freq[word])
        else:
            word_data.append(0)
            unknown_count += 1
    word_count[0][1] = unknown_count
    reversed_word_freq = dict(zip(word_freq.values(),word_freq.keys()))
    return word_count, word_data, word_freq, reversed_word_freq


def compute_freq(word_count):
    count = np.array([freq for word,freq in word_count],dtype=np.float32)
    freq = count/np.sum(count)   # 计算词频
    freq = freq ** (3./4.)       # 词频变换
    return count, freq


class SkipGram_Data(Dataset):
    def __init__(self, training_label, word_to_idx, idx_to_word, word_freqs):
        super(SkipGram_Data, self).__init__()
        self.text_encoded = torch.Tensor(training_label).long()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        idx = min( max(idx,C),len(self.text_encoded)-2-C) #防止越界
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx-C, idx)) + list(range(idx+1, idx+1+C))
        pos_words = self.text_encoded[pos_indices] 
        #多项式分布采样，取出指定个数的高频词
        neg_words = torch.multinomial(self.word_freqs, num_sampled+2*C, False)#True)
        #去掉正向标签
        neg_words = torch.Tensor(np.setdiff1d(neg_words.numpy(),pos_words.numpy())[:num_sampled]).long()
        return center_word, pos_words, neg_words


class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGram_Model, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        initrange = 0.5 / self.embed_size
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        input_embedding = self.in_embed(input_labels)
        pos_embedding = self.in_embed(pos_labels)
        neg_embedding = self.in_embed(neg_labels)
        log_pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze()
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze()
        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)
        loss = log_pos + log_neg
        return -loss



if __name__ == "__main__":

    all_words_list = []
    all_words_list = get_all_words(trainset,all_words_list)
    all_words_list = get_all_words(valset,all_words_list)
    all_words_list = get_all_words(testset,all_words_list)

    word_count, word_data, word_freq, reversed_word_freq = build_dataset(all_words_list, len(all_words_list))
    count, freq = compute_freq(word_count)
    word_size = len(word_freq)

    print("生成 skip-gram 模型 ......")
    train_skipgram = SkipGram_Data(word_data, word_freq, reversed_word_freq, freq)
    skipgram_loader = torch.utils.data.DataLoader(train_skipgram, batch_size=BATCH_SIZE,drop_last=True, shuffle=True)
    sample = iter(skipgram_loader)		# 将数据集转化成迭代器
    center_word, pos_words, neg_words = sample.next()	  # 从迭代器中取出一批次样本			
    #print(center_word[0], reversed_word_freq[np.compat.long(center_word[0])], [reversed_word_freq[i] for i in pos_words[0].numpy()])

    model = SkipGram_Model(word_size, EMBEDDING_SIZE).to(device)
    model.train()

    valid_size = 32
    valid_window = word_size/2  # 取样数据的分布范围.
    valid_examples = np.random.choice(int(valid_window), valid_size, replace=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    NUM_EPOCHS = 10

    for e in range(NUM_EPOCHS):
        for ei, (input_labels, pos_labels, neg_labels) in enumerate(skipgram_loader):
            input_labels = input_labels.to(device)
            pos_labels = pos_labels.to(device)
            neg_labels = neg_labels.to(device)
            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()
            optimizer.step()
            if ei % 20 == 0:
                print("epoch: {}, iter: {}, loss: {}".format(e, ei, loss.item()))

        if e % 40 == 0:           
            norm = torch.sum(model.in_embed.weight.data.pow(2),-1).sqrt().unsqueeze(1)
            normalized_embeddings = model.in_embed.weight.data / norm
            valid_embeddings = normalized_embeddings[valid_examples]
            similarity = torch.mm(valid_embeddings, normalized_embeddings.T)
            for i in range(valid_size):
                valid_word = reversed_word_freq[valid_examples[i]]
                top_k = 8  # 取最近的排名前8的词
                nearest = (-similarity[i, :]).argsort()[1:top_k + 1]  #argsort函数返回的是数组值从小到大的索引值
                log_str = 'Nearest to %s:' % valid_word  
                for k in range(top_k):
                    close_word = reversed_word_freq[nearest[k].cpu().item()]
                    log_str = '%s,%s' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings

    with open('skip-gram-model.txt', 'a') as f:    
        for i in range(len(reversed_word_freq)):
            f.write(reversed_word_freq[i] + str(list(final_embeddings.numpy()[i])) + '\n')
    f.close()
    print("skip-gram 词向量生成完毕！")
