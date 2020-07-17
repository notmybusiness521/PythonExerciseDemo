# --*--coding=utf8--*--
'''
@Author:
@Time:
@Describe:
'''
import tensorflow as tf
from tensorflow import feature_column as fc
from tensorflow.python.feature_column.feature_column import _LazyBuilder
import warnings
warnings.filterwarnings("ignore")

def transform_fn(x):
    return x + 2


price = {'price': [[1.], [2.], [3.], [4.]]}
# builder = _LazyBuilder(price)
price_column = fc.numeric_column('price', normalizer_fn=transform_fn)
# price_transformed_tensor = price_column._get_dense_tensor(builder)
#
# with tf.Session() as session:
#     print(session.run([price_transformed_tensor]))

price_transformed_tensor = fc.input_layer(price, [price_column])

with tf.Session() as session:
    print(session.run([price_transformed_tensor]))


