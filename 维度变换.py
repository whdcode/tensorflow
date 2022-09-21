"""tensorflow三大核心操作之一"""

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
outline:
▪ shape, ndim
▪ reshape
▪ expand_dims/squeeze
▪ transpose
▪ broadcast_to
"""

"""
temsorflow 的content是（b, h, w, c)
1、veiw
[b, 28, 28]
[b, 28 * 28]
[b, 2, 14 * 28]
[b, 28, 28, 1]
"""

"""reshape()  不更改数据的情况下为数组赋予新形状
tf.reshape(tensor, shape, name=None)
"""
# 实例,保证数据size前后不变且具有物理意义
a = tf.random.normal([4, 28, 28, 3])
a1 = tf.reshape(a, [4, 784, 3]).shape     # 保持原数据不变，28 * 28 = 784
a2 = tf.reshape(a, [4, -1, 3]).shape      # 依据形状变换不改变数据前提，会自动推出-1的值为784
a3 = tf.reshape(a, [4, 2352]).shape
print(a1, a2, a3)

"""tf.transpose()来变换维度
tf.transpose(tensor, [变换后的维度列表,若不指定则默认进行转置操作])
"""
b = tf.random.normal([4, 28, 28, 3])
print(b.shape)
b1 = tf.transpose(b, [3, 2, 1, 0])
print(b1.shape)
b2 = tf.transpose(b)    # 默认转置
print(f"tf.transpose{b2.shape}")

"""维度的压缩和扩展"""
"""1、维度的扩展tf.expend_dims(tensor, axis=所增加维度在新tensor上的位置）
axis为负值从反方向计数新维度的位置"""
# 示例
c = tf.expand_dims(b, axis=1)
print(c.shape)
c1 = tf.expand_dims(b, axis=-2)
print(c1.shape)
"""2、维度的减少tf.squeeze(tensor, axis=要消除的维度）不指定axis时默认去掉所有为1的维度
例如[1,2,3,2] = [2,3,2]
[1,2,3,1] = [2,3]
"""
# 示例
d = tf.zeros([1, 2, 3, 1, 1])
d1 = tf.squeeze(d).shape
print(d1)
d2 = tf.squeeze(d, 0).shape
print(d2)
d3 = tf.squeeze(d, -2).shape
print(d3)


"""tf.broadcast广播, 一种计算运行的优化手段，没有扩张数据，多维数组计算时自动进行
过程:

①右边对齐，小维度对齐    [4,32,14,14]   >>>  ②维度插入   [4,32,14,14]   >>>  ⑥维度值为1的维度进行拓展（初始维度必须为1） [4,32,14,14]
                      [         2]                   [1,1,1,2]                                               [4,32,14,2]   
  
"""
# 示例
e = tf.zeros([2, 2, 3, 2, 2])
e1 = (e + d).shape
print(e1)

# 显式进行广播，扩展到想要的shape
e2 = tf.broadcast_to(d, [3, 2, 3, 2, 2])
print(e2.shape)
e3 = tf.broadcast_to(e, [2, 2, 2, 3, 2, 2])
print(e3.shape)

# 对比broadcast_to 和tf.tile知，前者是一种计算优化，比后者占用更少的内存
f = tf.ones([2, 3])
f1 = tf.broadcast_to(f, [2, 2, 3])
print(f"f1: {f1.shape}")

f2 = tf.expand_dims(f, axis=0)
f3 = tf.tile(f2, [2, 1, 1])
print(f"f3:{f3.shape}")