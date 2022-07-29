import torch
import torch.nn as nn


# 使用EcoderLayer类来实现编码器
from MultiHeadedAttention import clones
from SubLayerConnection import SubLayerConnection


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        :param size: 词嵌入的维度
        :param self_attn: 自注意力实例
        :param feed_forward: 前馈全连接层的实例
        :param dropout: 网络置零比率
        """
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # 每层有两个子结构，所以需要两个sublayer
        self.sublayer = clones(SubLayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        :param x: 上一层的输出
        :param mask: 掩码张量
        """
        # 第一层为多头注意力层，第二层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)