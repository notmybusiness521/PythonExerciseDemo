# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:
'''

import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("train_path", "/path/to/test/*", "training data dir")
tf.app.flags.DEFINE_integer("train_steps", 1000, "total train steps")

if __name__ == '__main__':
    train_path = FLAGS.train_path
    print(train_path)

    steps = FLAGS.train_steps
    print(steps * 3)