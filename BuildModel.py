import copy

import torch.nn as nn
import torch
from pyitcast.transformer_utils import Batch
# 导入优化器工具包get_std_opt, 该工具用于获得标准的针对Transformer模型的优化器
# 该标准优化器基于Adam优化器, 使其对序列到序列的任务更有效.
from pyitcast.transformer_utils import get_std_opt
# 导入标签平滑工具包, 该工具用于标签平滑, 标签平滑的作用就是小幅度的改变原有标签值的值域
# 因为在理论上即使是人工的标注数据也可能并非完全正确, 会受到一些外界因素的影响而产生一些微小的偏差
# 因此使用标签平滑来弥补这种偏差, 减少模型对某一条规律的绝对认知, 以防止过拟合. 通过下面示例了解更多.
from pyitcast.transformer_utils import LabelSmoothing
# 导入损失计算工具包, 该工具能够使用标签平滑后的结果进行损失的计算,
# 损失的计算方法可以认为是交叉熵损失函数.
from pyitcast.transformer_utils import SimpleLossCompute
import numpy as np
from torch.autograd import Variable

from Decoder import Decoder
from DecoderLayer import DecoderLayer
from Embeddings import Embeddings
from EncoderLayer import EncoderLayer
from Ecoder import Encoder
from EncoderDecoder import EncoderDecoder
from Generator import Generator
from MultiHeadedAttention import MultiHeadedAttention
from PositionEmbedding import PositionEmbedding
from PositionwiseFeedForward import PositionwiseFeedForward


def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    """该函数用来构建模型, 有7个参数，分别是源数据特征(词汇)总数，目标数据特征(词汇)总数，
           编码器和解码器堆叠数，词向量映射维度，前馈全连接网络中变换矩阵的维度，
           多头注意力结构中的多头数，以及置零比率dropout."""

    # 获得一个深度拷贝，对模型结构进行深度拷贝，使他们彼此独立，不受干扰。
    c = copy.deepcopy

    # 实例化多头注意力类，得到对象attn
    attn = MultiHeadedAttention(head, d_model)

    # 实例化前馈全连接类
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # 实例化位置编码类，得到对象postion
    postion = PositionEmbedding(d_model, dropout)

    # 根据结构图, 最外层是EncoderDecoder，在EncoderDecoder中，
    # 分别是编码器层，解码器层，源数据Embedding层和位置编码组成的有序结构，
    # 目标数据Embedding层和位置编码组成的有序结构，以及类别生成器层.
    # 在编码器层中有attention子层以及前馈全连接子层，
    # 在解码器层中有两个attention子层以及前馈全连接层.
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(postion)),
        nn.Sequential(Embeddings(d_model, source_vocab), c(postion)),
        Generator(d_model, target_vocab)
    )

    # 模型结构完成后，接下来就是初始化模型中的参数，比如线性层中的变换矩阵
    # 这里一但判断参数的维度大于1，则会将其初始化成一个服从均匀分布的矩阵，
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal(p)

    return model


def data_generator(V, batch, num_batch):
    """
    该函数用于随机生成copy任务的数据，它的三个输入函数中V是代表随机生成的数字中最大值加1，
    batch：每次输送给模型更新一次参数的数据量，num_batch：一共输送num_batch次完成一轮
    """

    # 使用for循环遍历num_batch次
    for i in range(num_batch):
        # 使用np中的rand.randint()随机生成[1, V)范围的整数
        # 分布在(batch, 10)形状的矩阵当中，然后再把numpy形式转换成torch中的tensor
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))

        # 把矩阵当中的第一列置1，使这一列为起始标志列
        # 当解码器进行一次解码的时候，会使用起始标志作为输入
        data[:, 0] = 1

        # 因为是copy任务，所有source和target是完全相同的，且数据样本作为变量不需要求梯度
        # 因此require_grad设置为false
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)

        # 使用Batch对source和target进行对应批次的掩码张量生成，最后使用yield返回
        yield Batch(source, target)



if __name__ == '__main__':
    # source_vocab = 11
    # target_vocab = 11
    # N=6
    # res = make_model(source_vocab, target_vocab)
    # print(res)

    # 生成1至10的随机数
    V = 11

    # 每次给模型20个样本进行参数学习
    batch = 20

    # 连续学习30也就是给模型整30次，没次给20个样本的学习。30次刚好完成1轮。
    num_batch = 30

    res = data_generator(V, batch, num_batch)

    # 使用make_model获得模型
    model = make_model(V, V, N=2)

    # 使用get_std_opt获得模型优化器
    model_optimizer = get_std_opt(model)

    # 使用LabelSmoothing获得标签平滑对象
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)

    # 使用SimpleLossCompute获得利用标签平滑结果的损失计算方法
    loss = SimpleLossCompute(model.generator, criterion, opt=model_optimizer)