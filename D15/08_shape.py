"""
Tensor shape change
Static shape
Dynamic shape
"""

import tensorflow as tf

plhd = tf.placeholder('float32', [None, 3])
# Static shape
print(plhd)
plhd.set_shape([4, 3])
print(plhd)
# Dynamic shape
new_plhd = tf.reshape(plhd, shape=[1, 3, 4])
print(new_plhd)
new_plhd = tf.reshape(plhd, shape=[1, 1, 3, 4])
print(new_plhd)

with tf.Session() as sess:
    pass