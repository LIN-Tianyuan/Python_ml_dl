"""
Create Tensor
"""
import tensorflow as tf
import numpy as np

tensor1d = tf.constant([1, 2, 3, 4, 5])
tensor2d = tf.constant(np.arange(1, 7).reshape(2, 3))
tensor = tf.constant(100.0, shape=(2, 3))

tensornd = tf.random_normal(shape=(5, 5),
                            mean=1.80,
                            stddev=0.2,
                            dtype='float32')

zeros = tf.zeros(shape=(2, 3))
ones = tf.ones(shape=(2, 3))
zeros_like = tf.zeros_like(tensor1d)


with tf.Session() as sess:
    print(tensor1d.eval())
    print(tensor2d.eval())
    print(tensor.eval())
    print(tensornd.eval())
    print(zeros.eval())
    print(ones.eval())
    print(zeros_like.eval())