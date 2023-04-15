import torch
import torch.nn as nn


class TextCNN_Model(nn.Module):

    def __init__(self, configs,pre_embedding):
        super(TextCNN_Model, self).__init__()

        vocab_size = configs.vocab_size
        embedding_dimension = configs.embedding_dimension
        label_num = configs.label_num

        # 词嵌入和dropout
        self.pre_embed = nn.Embedding.from_pretrained(pre_embedding,freeze=False)
        self.sent_embed = nn.Embedding(vocab_size+2, embedding_dimension)
        self.pos1_embed = nn.Embedding(153, 10)
        self.pos2_embed = nn.Embedding(153, 10)
        self.dropout = nn.Dropout(configs.dropout)
        self.covns = nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels=embedding_dimension+20,
									out_channels=128,
									kernel_size=k),nn.Tanh(),nn.MaxPool1d(kernel_size=98-k+1)) for k in range(2,6)])
        self.linear = nn.Linear(128*4,label_num)

    def forward(self, x):
        sent = x['text']
        pos1 = x['pos1']
        pos2 = x['pos2']        
        sent_embed = self.pre_embed(sent)
        
        pos1_embed = self.pos1_embed(pos1)
        pos2_embed = self.pos2_embed(pos2)

        input = torch.cat([sent_embed,pos1_embed,pos2_embed],dim=2) #[64,100,120]
        input = input.permute(0,2,1) #[64,120,100]
        input = self.dropout(input)
        input = [conv(input) for conv in self.covns] #4
        output = torch.cat(input,dim=1) #[64,512,1]
        output = self.dropout(output)
        output = output.view(-1,output.size(1)) #[64,512]
        output = self.dropout(output)
        output = self.linear(output) #[64,44]
        return output
    


