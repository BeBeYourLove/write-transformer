import torch
import torch.nn as nn

from MultiHeadedAttention import clones
from SubLayerConnection import SubLayerConnection


# 使用DecoderLayer的类实现解码器层
class DecoderLayer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        :param size: 词嵌入的维度，也代表解码器的尺寸
        :param self_attn: 多头自注意力对象，也就是Q=K=V
        :param src_attn: 多头注意力对象，也就是Q!=K=V
        :param feed_forward: 前馈全连接对象
        :param dropout: 置0比率
        """
        super(DecoderLayer, self).__init__()
        # 在初始化函数中，主要就是将这些传输到类中
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 使用clones函数克隆三个子层连接对象
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        """
        :param x: 来自上一层的输入的x
        :param memory: 编码器的输出张量
        :param source_mask: 源数据的掩码张量
        :param target_mask: 目标数据的掩码张量
        """

        m = memory

        # 将x传入第一个子层结构，第一层子层结构的输入是x和self-attn函数，因为是自注意力机制，所以Q,K,V都是x
        # 最后一个参数是目标数据掩码张量，这时要对目标数据进行遮掩，因为此时模型可能还没有生成任何目标数据，
        # 比如在解码器准备生成第一个字符或词汇时，我们其实已经传入了第一个字符以便计算损失，
        # 但是我们不希望在生成第一个字符时模型能利用这个信息，因此我们会将其遮掩，同样生成第二个字符或词汇时，
        # 模型只能使用第一个字符或词汇信息，第二个字符以及之后的信息都不允许被模型使用.
        x = self.sublayer[0](x, lambda x:self.self_attn(x, x, x, target_mask))

        # 接着进入第二个子层，这个子层中常规的注意力机制，q是输入x; k，v是编码层输出memory，
        # 同样也传入source_mask，但是进行源数据遮掩的原因并非是抑制信息泄漏，而是遮蔽掉对结果没有意义的字符而产生的注意力值，
        # 以此提升模型效果和训练速度. 这样就完成了第二个子层的处理.
        x = self.sublayer[1](x, lambda x:self.src_attn(x, m, m, source_mask))

        # 最后是一个子层就是前馈全连接子层，经过它的处理后就可以返回结果。
        return self.sublayer[2](x, self.feed_forward)