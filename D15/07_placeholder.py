"""
placeholder
"""
import numpy as np
import tensorflow as tf

x = tf.placeholder('float32', [None, 3])    # N, 3

with tf.Session() as sess:
    data = np.arange(1, 7).reshape(2, 3)
    res = sess.run(x, feed_dict={x:data})
    print(res)