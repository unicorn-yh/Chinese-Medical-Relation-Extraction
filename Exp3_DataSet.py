import torch
from torch.utils.data import Dataset


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


if __name__ == "__main__":
    trainset = TextDataSet(filepath="./data/data_train.txt")
    testset = TestDataSet(filepath="./data/test_exp3.txt")
    print("训练集长度为：", len(trainset))
    print("测试集长度为：", len(testset))
