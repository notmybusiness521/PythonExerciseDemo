# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:
'''
import os
import tensorflow as tf
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
def add():
    v_1 = tf.constant([1, 2, 3, 4])
    v_2 = tf.constant([2, 1, 5, 3])
    v_add = tf.add(v_1, v_2)
    with tf.Session() as sess:
        print(sess.run(v_add))
def tensor():
    #常量
    t1 = tf.constant(2)
    #张量
    t_zeros = tf.zeros([3, 2], tf.float32)
    ones_t = tf.ones([2, 3], tf.int32)
    #在一定范围内生成一个从初值到终值等差排布的序列
    range_t = tf.linspace(2.0, 5.0, 5)
    range_t1 = tf.range(10)
    #随机序列或张量
    #创建一个具有一定均值（默认值=0.0）和标准差（默认值=1.0）、形状为 [M，N] 的正态分布随机数组
    t_random = tf.random_normal([2, 3],mean=2.0, stddev=1.0, seed=12)
    #创建一个具有一定均值（默认值 = 0.0）和标准差（默认值 = 1.0）、形状为[M，N] 的截尾正态分布随机数组
    t_random1 = tf.truncated_normal([1, 5], stddev=2, seed=12)
    #要在种子的 [minval（default=0），maxval] 范围内创建形状为 [M，N] 的给定伽马分布随机数组
    t_random2 = tf.random_uniform([2, 3], maxval=4, seed=12)
    #要将给定的张量随机裁剪为指定的大小
    t_random3 = tf.random_crop(t_random, [2, 1], seed=12)
    with tf.Session() as sess:
        name = ["t1", "t_zeros", "ones_t", "range_t", "range_t1", "t_random", "t_random1", "t_random2", "t_random3"]
        res = sess.run([t1, t_zeros, ones_t, range_t, range_t1, t_random, t_random1, t_random2, t_random3])
        for i in range(len(res)):
            print(name[i], "  is:  \n", res[i])
def varianle_place():
    x = tf.placeholder("float")
    y = 2*x
    data = tf.random_uniform([4, 5], 10, seed=0)
    with tf.Session() as sess:
        x_data = sess.run(data)
        print(sess.run(y, feed_dict={x:x_data}))

def device():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.device('/cpu:0'):
        rand_t = tf.random_uniform([50, 50], 0, 10, dtype=tf.float32, seed=0)
        a = tf.Variable(rand_t)
        b = tf.Variable(rand_t)
        c = tf.matmul(a, b)
        init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)
    print(sess.run(c))


if __name__ == '__main__':
    # add()
    # tensor()
    # varianle_place()
    device()