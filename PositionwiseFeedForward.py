import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseEmbedding(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        :param d_model: 线性层输入维度
        :param d_ff: 线性层输出维度
        :param dropout: 随机置0比率
        """
        super(PositionwiseEmbedding, self).__init__()

        # 设置两个线性层分别为fc1, fc2
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        :param x: 来自上一层的输出
        :return: 本层线性层处理过之后的输出
        """
        res = self.fc1(x)
        res = F.relu(res)
        res = self.dropout(res)
        res = self.fc2(res)

        return res