# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:第一个测试程序
'''
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
message = tf.constant("Welcom to the exciting world of deep learning!")
with tf.Session() as sess:
    print(sess.run(message).decode())

