import torch
import torch.nn as nn

# 使用SubLayerConnection类来实现子层连接结构
from LayerNorm import LayerNorm


class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        """
        :param size: 词向量的维度
        :param dropout: 网络中置零比率
        """
        super(SubLayerConnection, self).__init__()
        # 先实例化LayerNorm
        self.norm = LayerNorm(size)
        # 实例化dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """
        :param x: 为上一层或子层的一个输出
        :param sublayer: 该层所要进行操作的网络
        """

        # 首先对输入进行规范化，然后传给子层处理，之后再对其尽心dropout
        # 防止过拟合，最后进行相加。将输入的x与子层处理之后的dropout结果进行相加作为最终子层的连接输出。
        return x + self.dropout(sublayer(self.norm(x)))
