import torch
from pyitcast.transformer_utils import LabelSmoothing
import matplotlib.pyplot as plt
from torch.autograd import Variable

# 使用LabelSmoothing实例化一个crit对象.
# 第一个参数size代表目标数据的词汇总数, 也是模型最后一层得到张量的最后一维大小
# 这里是5说明目标词汇总数是5个. 第二个参数padding_idx表示要将那些tensor中的数字
# 替换成0, 一般padding_idx=0表示不进行替换. 第三个参数smoothing, 表示标签的平滑程度
# 如原来标签的表示值为1, 则平滑后它的值域变为[1-smoothing, 1+smoothing].
crit = LabelSmoothing(size=5, padding_idx=0, smoothing=0.5)

# 标签的表示值是0，1，2
predict = Variable(torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                     [0, 0.2, 0.7, 0.1, 0],
                                     [0, 0.2, 0.7, 0.1, 0]]))

# 标签的表示值是0, 1, 2
target = Variable(torch.LongTensor([2, 1, 0]))

# 将predict, target传入对象中
crit(predict, target)

# 绘制标签平滑图像
plt.imshow(crit.true_dist)
plt.show()