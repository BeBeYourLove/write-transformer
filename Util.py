import numpy as np
import torch
import matplotlib.pyplot as plt

def subsequent_mask(size):
    """生成一个向后遮掩的掩码张量，参数size是掩码张量最后的两个维度，它的最后两个维度生成一个方阵"""
    # 首先定义掩码张量的形状
    attn_shape = (1, size, size)

    # 向矩阵中填充1形成上三角阵，其中的数据类型使用uint8
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 最后将numpy类型转化为torch中的tensor, 内部做一个1 - 的操作,
    # 在这个其实是做了一个三角阵的反转, subsequent_mask中的每个元素都会被1减,
    # 如果是0, subsequent_mask中的该位置由0变成1
    # 如果是1, subsequent_mask中的该位置由1变成0
    return torch.from_numpy(1 - subsequent_mask)

# size = 5
# sm = subsequent_mask(size)
# print(sm)

plt.figure(figsize=(5, 5))
plt.imshow(subsequent_mask(20)[0])
plt.show()
