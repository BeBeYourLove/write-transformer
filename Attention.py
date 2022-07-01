import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from Embeddings import Embeddings
from PositionEmbedding import PositionEmbedding


def attention(query, key, value, mask=None, dropout=None):
    # 在函数中先去到query最后一维的大小，通常是我们词嵌入的维度
    d_k = query.size(-1)

    # 按照自注意力的公式，将query和key的最后两个维度的转置进行相乘，再除以缩放系数根号下d_k.
    # 得到注意力得分张量scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 判断是否使用掩码张量
    if mask is not None:
        # 使用mask_fill方法进行掩盖，将tensor中对应为0的值替换为-1e9
        scores = scores.masked_fill(mask == 0, -1e9)

    # 对scores的最后一维进行softmax操作，使用F.softmax方法，方法的第一个参数是softmax对象，第二个参数是目标维度
    # 这样就获得了注意力张量
    p_attn = F.softmax(scores, dim=-1)

    # 之后判断是否使用dropout
    if dropout is not None:
        # 将p_attn传入dropout进行'丢弃'处理
        p_attn = F.dropout(p_attn)

    # 最后返回将p_attn与value相乘获得最终的query注意力表示，同时返回注意力张量
    return torch.matmul(p_attn, value), p_attn


# d_model = 512
# dropout = 0.1
# max_len = 60
# vocab = 1000
# embed = Embeddings(d_model, vocab)
# ten = torch.tensor([[234, 123, 543, 122], [235, 124, 789, 567]])
# x = embed(ten)
# pe = PositionEmbedding(d_model, dropout, max_len)
# res = pe(x)
#
# query = key = value = res
# mask = Variable(torch.zeros(2, 4, 4))
# attn, p_attn = attention(query, key, value, mask)
# print("attn:", attn)
# print("p_attn:", p_attn)