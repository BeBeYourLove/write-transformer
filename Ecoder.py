import torch
import torch.nn as nn

from LayerNorm import LayerNorm
from MultiHeadedAttention import clones

# 使用Ecoder类来进行编码器的实现
class Encoder(nn.Module):
    def __init__(self, layer, n):
        """
        :param layer: 编码器层
        :param n: 编码器层数
        """
        super(Encoder, self).__init__()
        # 使用clones函数对编码器层进行克隆，并放入self.layers中
        self.layers = clones(layer, n)
        # 再初始化一个规范层，它将用在编码器的最后面
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        :param x: x表示上一层的输出
        :param mask: 代表掩码张量
        :return:
        """
        # 首先对这个克隆编码器层进行循环，每次都会得到一个新的x,
        # 循环过程，就相当于输出的x经过N个编码器层的处理。
        # 最后通过规范层的对象的self.norm进行处理，最后返回结果。
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)