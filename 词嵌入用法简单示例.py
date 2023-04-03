"""
该文件旨在示意nn.Embedding函数的使用方式，以此来为实验开展做出
此代码已发布在CSDN上
"""

import torch
import torch.nn as nn

instr = "词嵌入怎么用？"
print("原句子为：", instr)

# 该查找表即为极简词汇表(vocabulary)
lookup_table = list("啊窝饿词入嵌怎用么？")
print("词汇表（极简版）为：", lookup_table)

inp = []
for ch in instr:
    inp.append(lookup_table.index(ch))
print("经过查找表之后，原句子变为：", inp)

inp = torch.tensor(inp)
embedding_dim = 3
emb = nn.Embedding(
    num_embeddings=len(lookup_table),
    embedding_dim=embedding_dim)

print("最终结果：")
print(emb(inp))
print("词嵌入就是这样用的！")
