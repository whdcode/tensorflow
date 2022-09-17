"""创建tensor"""

"""创建tensor的方式有：
▪ from numpy, list
▪ zeros, ones
▪ fill
▪ random
▪ constant
▪ Application
"""
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 使用numpy创建tensor
arrA = tf.convert_to_tensor(np.ones([2, 3]), dtype=tf.float32)    # 这里的[2, 3]为shape
print(arrA)
arrA1 = tf.convert_to_tensor(np.zeros([2, 3]), dtype=tf.float32)
print(arrA)

# 使用list创建tensor
arrB = tf.convert_to_tensor([2, 3])     # 这里[2, 3]为一维数组，shape=(2,)
print(arrB)
# 列表含有浮点型数字，则对应tensor为浮点型tensor
arrB1 = tf.convert_to_tensor([2, 3.])
print(arrB1)

# 两行一列的数组
arrB2 = tf.convert_to_tensor([[2.], [3]])   # [[2.], [3]]逻辑上
print(arrB2)

# 使用tf.zeros([])创建tensor
zero1 = tf.zeros([])    # 为浮点0.0
print(zero1)

zero2 = tf.zeros([1])   # 这里的[1]为shape
print(zero2)

zero3 = tf.zeros([1, 2])
print(zero3)

zero4 = tf.zeros([3, 1, 2])
print(zero4)
# 使用tf.zeros_like(a).意味创建一个和a的shape一样的tensor,等价为tf.zeros(a.shape)
zero_like = tf.zeros_like(zero4)
print(zero_like)

# 同理和tf.zeros([])创建tensor的tf.ones()
# ...

# tf.fill(shape, num)来创建tensor
tensor1 = tf.fill([2, 2], 0)    # 创建一个所有元素都为0的shape= [2, 2]的tensor
print(tensor1)


# tf.random.normal(shape, mean=均值，stddev=标准差）  取值服从 正态分布
tensor2 = tf.random.normal([2, 3], mean=1, stddev=1)
print(tensor2)

# tf.random.truncated_normal(shape, mean=均值，stddev=标准差）生成的值大于平均值2个标准偏差的值则丢弃重新选择。

tensor3 = tf.random.truncated_normal([2, 2], mean=1, stddev=1)
print(tensor3)