import torch
import torch.nn as nn
import torch.nn.functional as functions
from transformers import BertTokenizer, BertModel

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
    

class SentenceRE(nn.Module):

    def __init__(self, configs):
        super(SentenceRE, self).__init__()

        self.pretrained_model_path = configs.pretrained_model_path or 'bert-base-chinese'
        self.embedding_dimension = configs.embedding_dimension
        self.dropout = configs.dropout
        self.class_num = configs.class_num

        self.bert_model = BertModel.from_pretrained(self.pretrained_model_path)

        self.dense = nn.Linear(self.embedding_dimension, self.embedding_dimension)
        self.drop = nn.Dropout(self.dropout)
        self.activation = nn.Tanh()
        self.norm = nn.LayerNorm(self.embedding_dimension * 3)
        self.hidden2tag = nn.Linear(self.embedding_dimension * 3, self.class_num)

    def forward(self, token_ids, token_type_ids, attention_mask, e1_mask, e2_mask):
        sequence_output, pooled_output = self.bert_model(input_ids=token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=False)

        # 每个实体的所有token向量的平均值
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        e1_h = self.activation(self.dense(e1_h))
        e2_h = self.activation(self.dense(e2_h))

        # [cls] + 实体1 + 实体2
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        concat_h = self.norm(concat_h)
        logits = self.hidden2tag(self.drop(concat_h))

        return logits

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector
