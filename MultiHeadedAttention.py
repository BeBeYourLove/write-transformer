import torch.nn as nn
import copy
from Attention import attention
from Embeddings import Embeddings
import torch
from torch.autograd import Variable

# 由于在多头注意力机制实现中会用到多个结构相同的线性层
# 所以使用clones函数将他们初始化在同一个网络层列表对象中，之后的结构也会用到该函数
from PositionEmbedding import PositionEmbedding


def clones(model, n):
    """model代表需要克隆的网络模型，n为克隆所需的数量"""
    # 对model进行深拷贝，使其成为独立的层。然后放在nn.ModuleList类型的列表当中
    return nn.ModuleList([copy.deepcopy(model) for _ in range(n)])

# 多头注意力机制类
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        """初始化需要三个参数，head代表头数，embedding_dim代表词嵌入的维度，dropout代表置0的比率，默认为0.1"""
        super(MultiHeadedAttention, self).__init__()

        # 这里需要判断词嵌入的维度是否可以被多头进行整除
        # 由于后面每个头都会分配等量的词特征，也就是embedding_dim/head个
        assert embedding_dim % head == 0

        # 得到每个头获得的分割词向量维度d_k
        self.d_k = embedding_dim // head

        # 传入头数
        self.head = head

        # 获得线性层对象通过nn.liner进行实例化，每个全连接层的内部变换矩阵为embedding_dim * embedding_dim，然后使用clones函数克隆4个
        # 克隆4个是由于除了query,key,value三个矩阵各需要一个全连接层外，最后拼接的矩阵还需要一个进行拼接。
        self.liners = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        # self.attn为None，它代表最后得到的注意力张量，初始化为None
        self.attn = None

        # 最后设置dropout对象
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):

        # 如果存在掩码张量mask
        if mask is not None:
            # 使用unsqueeze进行维度拓展
            mask = mask.unsqueeze(0)

        # 接着我们获取batch_size变量，他是query尺寸的第一个数字，代表有多少条样本。
        batch_size = query.size(0)

        # 之后进入真正的多头处理环节
        # 首先利用zip函数将q,k,v与各个线性层组合到一起，利用迭代将输入的q,k,v分别传到线性层中
        # 做完线性变换后，开始为每个头分割输入，这里使用view方法对线性变换的结果进行维度重塑，多加了一个维度h，代表头数，
        # 这样就意味着每个头可以获得一部分词特征组成的句子，其中的-1代表自适应维度，
        # 计算机会根据这种变换自动计算这里的值.然后对第二维和第三维进行转置操作，
        # 为了让代表句子长度维度和词向量维度能够相邻，这样注意力机制才能找到词义与句子位置的关系，
        # 从attention函数中可以看到，利用的是原始输入的倒数第一和第二维.这样我们就得到了每个头的输入。
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
             for model, x in zip(self.liners, (query, key, value))]

        # 得到了每个头的输入之后，直接将他们传入attention中，
        # 这里可以直接调用之前创建的attention函数，同时也将mask和dropout传入其中
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 通过多头注意力计算后，我们就得到了每个头计算结果组成的4维张量，我们需要将其转换为输入的形状以方便后续的计算，
        # 因此这里开始进行第一步处理环节的逆操作，先对第二和第三维进行转置，然后使用contiguous方法，
        # 这个方法的作用就是能够让转置后的张量应用view方法，否则将无法直接使用，
        # 所以，下一步就是使用view重塑形状，变成和输入形状相同。
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 最后使用使用线性层列表中的最后一层对输入进行线性变换得到最终的多头注意力的结构输出
        return self.liners[-1](x)


# head = 8
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
# mask = Variable(torch.zeros(8, 4, 4))
# mha = MultiHeadedAttention(head, embedding_dim, dropout)
# mha_res = mha(query, key, value, mask)
# print(mha_res)

