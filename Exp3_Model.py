import torch
import torch.nn as nn
import torch.nn.functional as functions


class TextCNN_Model(nn.Module):

    def __init__(self, configs):
        super(TextCNN_Model, self).__init__()

        vocab_size = configs.vocab_size
        embedding_dimension = configs.embedding_dimension
        label_num = configs.label_num

        # 词嵌入和dropout
        self.embed = nn.Embedding(vocab_size, embedding_dimension)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, sentences):
        print(sentences)
        return 0
