import torch
import PositionEmbedding
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

# 创建一张15 x 5大小的画布
plt.figure(figsize=(15, 5))

# 实例化PositionEmbedding类得到Pe对象
pe = PositionEmbedding.PositionEmbedding(20, 0)

# 然后向pe传入被Variable封装的tensor，这样pe会直接执行forward函数
# 且这个tensor力的数值都是0，被处理后相当于位置编码张量
y = pe(Variable(torch.zeros(1, 100, 20)))

# 然后定义画布的横纵坐标，横坐标到100的长度，纵坐标是某一个词汇中的某维特征在不同长度下对应的值
# 因为总共有20维之多，我们这里只查看了10,11,12,13维的值。
plt.plot(np.arange(100), y[0, :, 5:9].data.numpy())

# 在画布上填写维度提示信息
plt.legend(["dim %d"%p for p in [5, 6, 7, 8]])
plt.show()
