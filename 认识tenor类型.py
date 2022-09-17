"""认识tensor的类型"""


import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建一个常量

a = tf.constant(1)  # 整型
b = tf.constant(1.)     # 单精度浮点
c = tf.constant(1.2, dtype=tf.double)   # 指定双精度浮点
d = tf.constant([True, False])  # 布尔型
e = tf.constant('hello world')  # string型
print(e)


"""tensor属性"""

# 在cpu上创建tensor,只能在cpu上使用
with tf.device("cpu"):
    f = tf.constant([1])
# 在gpu上创建tensor，同理，，，
with tf.device("gpu"):
    g = tf.range(3)

# gpu 和cpu上的tensor相互转换

f_gpu = f.gpu()
g_cpu = g.cpu()

# 将tensor转换成numpy型
b.numpy()

# 获得tensor的维度,两种
b.ndim

tf.rank(b)

# 确定一个变量是否为tensor类型
tf.is_tensor(b)
isinstance(b, tf.Tensor)    # 不推荐

# 变量名.dtype，返回变量类型，如tf.float、tf.bool
a.dtype

"""转换成tensor"""
# 将h转换成tensor，两种
h = np.arange(3)    # 得到0-2的一维数组
hh = tf.convert_to_tensor(h, dtype=tf.int32)    # 得到32位的整型tensor的hh

h_h = tf.cast(h, dtype=tf.float32)  # 转换成32位float tensor

"""tf.variable()"""
j = tf.range(2)
jj = tf.Variable(j, name='input_data')  # jj具备两个属性，jj.name=input_data:0; jj.trainable=True

""""to numpy"""
k = tf.ones([]) # 标量型的tensor，值为1
# 将k转换成numpy型
k.numpy()
int(k)
float(k)


