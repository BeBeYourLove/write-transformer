import torch
import torch.nn as nn
import math
from torch.autograd import Variable

from Embeddings import Embeddings


class PositionEmbedding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        # d_model 词嵌入的维度 dropout 设置置0比率 max_len 每个句子的单词数
        super(PositionEmbedding, self).__init__()

        # 实例化dropout层，传入dropout后获得dropout对象
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个位置编码矩阵，它是一个0阵，矩阵维度为(max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # 初始化一个绝对位置矩阵， 这里直接使用单词在句子中的索引进行表示
        # 使用arange方法获得一个自然数序列向量，然后使用unsqueeze方法将其拓展向量维度使其成为矩阵
        # 之后position矩阵的维度就为(max_len, 1)/(5000, 1)
        position = torch.arange(max_len).unsqueeze(1)

        # 绝对位置初始化之后，接下来就是考虑如何将这些位置信息加入到位置编码矩阵中
        # 基本思路是将(max_len, 1)的绝对位置矩阵，变换为(max_len, d_model)的形状，然后覆盖原始的初始位置矩阵即可
        # 要做这种矩阵变换，就需要一个(1, d_model)形状的变换矩阵div_term，我们对这个变换矩阵的要求除了形状外
        # 还希望它能够将自然数的绝对位置编码缩放成足够小的数字，有助于在之后的梯度下降过程中更快的收敛.这样我们就可以开始初始化这个变换矩阵了.
        # 首先使用arange获得一个自然数矩阵， 但是细心的同学们会发现， 我们这里并没有按照预计的一样初始化一个1xd_model的矩阵，
        # 而是有了一个跳跃，只初始化了一半即1xd_model/2 的矩阵。 为什么是一半呢，其实这里并不是真正意义上的初始化了一半的矩阵，
        # 我们可以把它看作是初始化了两次，而每次初始化的变换矩阵会做不同的处理，第一次初始化的变换矩阵分布在正弦波上， 第二次初始化的变换矩阵分布在余弦波上，
        # 并把这两个矩阵分别填充在位置编码矩阵的偶数和奇数位置上，组成最终的位置编码矩阵.
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(1000.0)/d_model))
        # 奇数列上使用余弦变换
        pe[:, 0::2] = torch.sin(position * div_term)
        # 偶数列上使用正弦变换
        pe[:, 1::2] = torch.cos(position * div_term)

        # 这样我们就得到了位置编码矩阵pe, pe现在还只是一个二维矩阵(max_len, d_model)，要想和embedding的输出（一个三维张量）相加，
        # 就必须拓展一个维度，所以这里使用unsqueeze拓展维度.
        pe = pe.unsqueeze(0)

        # 最后把pe位置编码矩阵注册成模型的buffer，什么是buffer呢，
        # 我们把它认为是对模型效果有帮助的，但是却不是模型结构中超参数或者参数，不需要随着优化步骤进行更新的增益对象.
        # 注册之后我们就可以在模型保存后重加载时和模型结构与参数一同被加载.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 输入的x为句子里面单词词嵌入后的向量表示
        # 在相加之前我们对pe做一些适配工作，将这个三维张量的第二维也就是句子最大长度的那一维将切片到与输入的x的第二维相同即x.size(1)，
        # 因为我们默认max_len为5000一般来讲实在太大了，很难有一条句子包含5000个词汇，所以要进行与输入张量的适配.
        # 最后使用Variable进行封装，使其与x的样式相同，但是它是不需要进行梯度求解的，因此把requires_grad设置成false.
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)

        # 最后使用一层dropout，返回结果
        return self.dropout(x)

d_model = 512
dropout = 0.1
max_len = 60
vocab = 1000

embed = Embeddings(d_model, vocab)
ten = torch.tensor([[234, 123, 543, 122], [235, 124, 789, 567]])
x = embed(ten)
pe = PositionEmbedding(d_model, dropout, max_len)
res = pe(x)
print(res)
print(res.shape)