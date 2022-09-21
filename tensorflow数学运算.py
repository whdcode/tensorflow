"""
▪ +-*/
▪ **, pow, square
▪ sqrt
▪ //, %     //为求整运算
▪ exp, log
▪ @, matmul
▪ linear layer
"""
import tensorflow as tf
"""求整//"""
a = tf.fill([2, 3], 5)
b = tf.fill([2, 3], 3)
e = tf.ones([2, 3])
c = a // b  # 要求分子分母数据类型一致
d = a % b   # 要求分子分母为整形
print(c, d)

""""""

"""
tf.math.log()取自然对数e为底
若要取其他底的对数，则使用换底公式，logab = logb / loga
和tf.exp(x)为e的x次幂
"""
# 求log2..8
f = tf.math.log(8.) / tf.math.log(2.)   # 2的3次幂为8
print(f)


"""
求方计算:1、b**a  2、tf.pow(b, a)  都为b的a次方
求平方根：tf.sqrt(b)对b开根（b>0)且b为浮点型
"""

"""
矩阵相乘
1、@
2、tf.matmul(a, b)
"""
# 示例
h = tf.ones([4, 2, 3])
g = tf.fill([4, 3, 5], 2.)
n = tf.fill([2, 3], 1.)
nn = tf.broadcast_to(n, [4, 2, 3])
jj = nn@g
j = h@g     # h和g的后两维相乘
j1 = tf.matmul(h, g)
print(j.shape, j1.shape, jj.shape)

