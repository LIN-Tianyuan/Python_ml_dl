"""
Basic properties of tensor
"""
import tensorflow as tf

x = tf.constant(100.0, name='xxx')

with tf.Session() as sess:
    print(sess.run(x))

    print('name: ', x.name)
    print('dtype: ', x.dtype)
    print('shape: ', x.shape)
    print('graph: ', x.graph)
    print('op: ', x.op)