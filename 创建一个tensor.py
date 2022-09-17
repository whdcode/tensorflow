"""创建tensor"""
import keras.layers

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
arrA = tf.convert_to_tensor(np.ones([2, 3]), dtype=tf.float32)  # 这里的[2, 3]为shape
print(arrA)
arrA1 = tf.convert_to_tensor(np.zeros([2, 3]), dtype=tf.float32)
print(arrA)

# 使用list创建tensor
arrB = tf.convert_to_tensor([2, 3])  # 这里[2, 3]为一维数组，shape=(2,)
print(arrB)
# 列表含有浮点型数字，则对应tensor为浮点型tensor
arrB1 = tf.convert_to_tensor([2, 3.])
print(arrB1)

# 两行一列的数组
arrB2 = tf.convert_to_tensor([[2.], [3]])  # [[2.], [3]]逻辑上
print(arrB2)

# 使用tf.zeros([])创建tensor
zero1 = tf.zeros([])  # 为浮点0.0
print(zero1)

zero2 = tf.zeros([1])  # 这里的[1]为shape
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
tensor1 = tf.fill([2, 2], 0)  # 创建一个所有元素都为0的shape= [2, 2]的tensor
print(tensor1)

# tf.random.normal(shape, mean=均值，stddev=标准差）  取值服从 正态分布
tensor2 = tf.random.normal([2, 3], mean=1, stddev=1)
print(tensor2)

# tf.random.truncated_normal(shape, mean=均值，stddev=标准差）生成的值大于平均值2个标准偏差的值则丢弃重新选择。

tensor3 = tf.random.truncated_normal([2, 2], mean=1, stddev=1)
print(tensor3)

# numpy.random.uniform(shape, minval=最小值, maxval=最大值）. 函数原型：  numpy.random.uniform(low,high,size)
# 功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
tensor4 = tf.random.uniform([2, 2], minval=1, maxval=2)
print(tensor4)  # 结果为浮点型

# tf.range函数：创建数字序列
tensor5 = tf.range(2, 10, 2)
print(tensor5)  # [2 4 6 8]
# tf.random_shuffle 打散tensor第一维度
tensor6 = tf.random.shuffle(tensor5)
print(tensor6)  # [2 8 4 6]

# tf.gather,gather(需要切片的目标， 切片依据的索引)用于根据提供的索引对输入张量第一维进行切片
tensor7 = tf.gather(tensor6, [0, 2])  # [0, 2]为切片索引
print(tensor7)

tensor8 = tf.gather(arrA, [0])  # 根据索引[0]知为获取第一行切片
print(tensor8)

"""使用tf.constant"""
cons1 = tf.constant(1)  # 创建一个标量值为1
print(cons1)
cons2 = tf.constant([2])  # [2]， 为列表对象
print(cons2)
cons3 = tf.constant([2, 2.])  # [2,2.]， 为列表对象,返回一个列表型tensor，元素转换为float
print(cons3)
# 当转换对象为非矩阵式，则会报错
# cons4 = tf.constant([[1, 2], [3]])    # [[1, 2], [3]]为非矩形
# print(cons4)

# 独热编码，将input转化为one-hot类型数据输出：tf.one_hot([1, 2, 3, 4, 5, 6], depth=5, on_value=2， off_value=1)
# 超出depth-1的输出全为0
tensor9 = tf.one_hot([1, 2, 3, 4, 5, 6], depth=7, on_value=1)
print(tensor9)

# loss=tf.keras.losses.mse(y, out)  # 计算输出值和预测值的均方误差
out = tf.random.normal([6, 7])  # 模拟一个输出
loss = tf.keras.losses.mse(tensor9, out)  # 假设tensor9为对应的标签列表的独热编码，loss为一个shape=[6,]的tensor
print(loss)

# tf.reduce_mean(loss)
"""reduce_mean(input_tensor,
                axis=None,  # 设定第几维求均值
                keep_dims=False,    # 是否保持原来tensor维度
                name=None,
                reduction_indices=None)"""
# 求各层输出后的平均loss
loss_mean = tf.reduce_mean(loss)  # 输出为标量shape=()
print(loss_mean)

# np.shape(对象），返回该对象的shape属性，若为常数，则表示标量，返回（）

"""keras.layers.Dense（）方法--定义网络层
keras.layers.Dense(units, 
				  activation=None, 
				  use_bias=True, 
				  kernel_initializer='glorot_uniform', 
				  bias_initializer='zeros', 
				  kernel_regularizer=None, 
				  bias_regularizer=None, 
			      activity_regularizer=None, 
				  kernel_constraint=None, 
				  bias_constraint=None)
"""
# 定义一个含有10个神经元的神经网络层
net10 = keras.layers.Dense(10)
print(net10)

""".build()方法"""

"""
...
"""

"""dim = 3tensor: 自然语言处理
   dim = 4tensor: 卷积神经网络
   dim = 5tensor: meta-learning 元学习
"""