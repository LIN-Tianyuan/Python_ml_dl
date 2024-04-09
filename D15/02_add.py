"""
Tensor execution summation
"""
import tensorflow as tf

# Definition
x = tf.constant(100.0)
y = tf.constant(200.0)
temp = tf.add(x, y)

# Execution
with tf.Session() as sess:
    print(sess.run(temp))

