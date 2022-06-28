import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):

    # 初始化函数两个参数d_model为嵌入的维度， vocab为词表数量
    def __init__(self, d_model, vocab):
        # 使用super函数指明继承nn.Model的初始化函数
        super(Embeddings, self).__init__()
        # 获得词嵌入对象Embedding
        self.lut = nn.Embedding(vocab, d_model)
        # 最后将d_model传入对象中
        self.d_model = d_model

    # 向前传播的计算函数，向类中传入数据后会自动调用该对象的forward方法进行计算
    def forward(self, x):
        # 因为Embedding层是首层，所以代表输入给文本的模型的文本通过词汇映射之后的向量
        # 将x传给self.lut并与根号下self.d_model相乘作为结果返回
        return self.lut(x) * math.sqrt(self.d_model)

# # 词嵌入的维度为512
# d_model = 512
# # 词表大小定义为1000
# vocab = 1000
#
# embed = Embeddings(d_model, vocab)
# ten = torch.tensor([[234, 123, 543, 122], [235, 124, 789, 567]])
# print(embed(ten))