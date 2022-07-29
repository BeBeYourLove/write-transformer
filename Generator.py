import torch
import torch.nn as nn
import torch.nn.functional as F

# 将线性层和sofmax层结合在一起形成最后的的输出层结构
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        """
        d_model为词嵌入的维度，vocab_size代表词表大小。
        """
        super(Generator, self).__init__()
        # 定义一个线性层
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """前向逻辑函数中输入是上一层的输出张量x"""
        # 在函数中, 首先使用上一步得到的self.project对x进行线性变化,
        # 然后使用F中已经实现的log_softmax进行的softmax处理.
        # 在这里之所以使用log_softmax是因为和我们这个pytorch版本的损失函数实现有关, 在其他版本中将修复.
        # log_softmax就是对softmax的结果又取了对数, 因为对数函数是单调递增函数,
        # 因此对最终我们取最大的概率值没有影响. 最后返回结果即可.
        return F.log_softmax(self.project(x), dim=-1)

