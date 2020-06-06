# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:
'''

import tensorflow as tf
import numpy as np
# 设 m 是样本数量，n 是特征数量，P 是类别数量
m = 1000
n = 15
p = 2
#*********************************************************
# 在标准线性回归的情况下，只有一个输入变量和一个输出变量
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
# 权重
w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)
#
Y_hat = w0 + w1 * X
# 损失函数
loss = tf.square(Y - Y_hat, name="loss")
#*********************************************************
#在多元线性回归的情况下，输入变量不止一个，而输出变量仍为一个。现在可以定义占位符X的大小为 [m，n]，其中 m 是样本数量，n 是特征数量，代码如下：
X1 = tf.placeholder(tf.float32, shape=[m, n], name="X1")
Y1 = tf.placeholder(tf.float32, name="Y1")
w0 = tf.Variable(0.0)
w1 = tf.Variable(tf.random_normal([n, 1]))
Y1_hat = tf.matmul(X, w1) + w0


