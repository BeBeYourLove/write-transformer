import torch
import torch.nn as nn

from MultiHeadedAttention import clones
from LayerNorm import LayerNorm

# 使用类Decoder来实现解码器
class Decoder(nn.Module):
    def __init__(self, layer, N):
        """
        layer代表响应的解码层
        N代表解码器的个数
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):

        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)

