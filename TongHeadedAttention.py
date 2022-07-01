import torch.nn as nn
from Attention import attention
from Embeddings import Embeddings
import torch
import copy
from torch.autograd import Variable
from PositionEmbedding import PositionEmbedding

"""
由于觉得源实现过去简单，尝试自己觉得的多头机制。由于计算限制这里只使用三个头
"""

def clones(model, n):
    return nn.ModuleList([copy.deepcopy(model) for _ in range(n)])

class TongHeadedAttention(nn.Module):

    def __init__(self, head, embedding_dim, dropout=0.1):

        super(TongHeadedAttention, self).__init__()
        self.head = head
        self.liners = clones(nn.Linear(512, 512), 9)
        self.fcend = nn.Linear(1536, 512)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.querys = []
        self.keys = []
        self.values = []
        # self.x = []

    def forward(self, query, key, value, mask=None):

        if mask is not None:
            mask = mask.unsqueeze(0)

        batch_size = query.size(0)

        # 获得三头query矩阵，并存入query列表
        query1 = self.liners[0](query)
        query2 = self.liners[1](query)
        query3 = self.liners[2](query)
        self.querys.append(query1)
        self.querys.append(query2)
        self.querys.append(query3)

        # 获得三头key矩阵，并存入key列表
        key1, key2, key3 = \
        [self.liners[i](key) for i in (3, 4, 5)]
        self.keys.append(key1)
        self.keys.append(key2)
        self.keys.append(key3)

        # 获得三头value矩阵，并存入value列表
        value1, value2, value3 = \
        [self.liners[i](value) for i in (6, 7, 8)]
        self.values.append(value1)
        self.values.append(value2)
        self.values.append(value3)


        # for i in range(0, 1, 2):
        #     res = None
        #     res, _ = attention(self.querys[i], self.keys[i], self.values[i], mask=mask, dropout=self.dropout)
        #     self.x.append(res)
        x1, _ = attention(self.querys[0], self.keys[0], self.values[0], mask=mask, dropout=self.dropout)
        x2, _ = attention(self.querys[1], self.keys[1], self.values[1], mask=mask, dropout=self.dropout)
        x3, _ = attention(self.querys[2], self.keys[2], self.values[2], mask=mask, dropout=self.dropout)

        res = torch.cat([x1, x2, x3], dim=-1)

        mlt_res = self.fcend(res)

        return mlt_res.squeeze(0)


# head = 3
# embedding_dim = 512
# d_model = 512
# dropout = 0.1
# max_len = 60
# vocab = 1000
# embed = Embeddings(d_model, vocab)
# ten = torch.tensor([[234, 123, 543, 122], [235, 124, 789, 567]])
# x = embed(ten)
# pe = PositionEmbedding(d_model, dropout, max_len)
# res = pe(x)
# query = key = value = res
# print("query等输入的维度为:", query.shape)
# mask = Variable(torch.zeros(2, 4, 4))
# tongAttention = TongHeadedAttention(head, embedding_dim, dropout)
# mha_res= tongAttention(query, key, value, mask)
# print(mha_res.shape)
# print(mha_res)