import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""1、合并"""
# tf.concat拼接,除了拼接的维度可以不等，其他维度都要相等
a = tf.ones([4, 54, 5])
b = tf.ones([2, 54, 5])
a1 = tf.ones([4, 2, 3])
a2 = tf.ones([4, 2, 3])
c = tf.concat([a, b], axis=0)
c2 = tf.concat([a1, a2], axis=1)
print("concat:", c.shape, c2.shape)  # concat: (6, 54, 5) (4, 4, 3)

# tf.stack堆叠，创造一个新维度，需要碓叠的两个tensor所有维度都要相等
d = tf.stack([a1, a2], axis=0)
d2 = tf.stack([a1, a2], axis=3)
print("stack:", d.shape, d2.shape)  # stack: (2, 4, 2, 3) (4, 2, 3, 2)
"""2、 分割"""

# tf.unstack,axis指定的维度有多少就拆成几个tensor
e, e1, e2, e3 = tf.unstack(d2, axis=0)  # d2 = (4, 2, 3, 2)
f = tf.unstack(d2, axis=0)
print("unstack:", e.shape, e1.shape, e2.shape, e3.shape, len(f))

# tf.split
g = tf.ones([8, 24, 48, 6])
g1 = tf.split(g, axis=0, num_or_size_splits=[2, 2, 4])  # 按照2：2：4比例分割第一维.返回列表
g2 = tf.split(g, axis=0, num_or_size_splits=4)      # 均分分成四个tensor，返回一个列表
print("split", g1[0].shape, g1[1].shape, g1[2].shape, g2[0].shape)
