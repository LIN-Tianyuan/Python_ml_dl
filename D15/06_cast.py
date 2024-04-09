"""
Type conversion of tensor
"""
import tensorflow as tf

tensor = tf.ones(shape=(10, ), dtype='bool')
temp = tf.cast(tensor, 'float32')

with tf.Session() as sess:
    print(sess.run(temp))
