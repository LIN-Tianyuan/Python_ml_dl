"""
The first piece of Tensorflow code
version : 1.14
"""

import tensorflow as tf

# Definition
hello = tf.constant('hello world')
sess = tf.Session()

# Execution
res = sess.run(hello)
print(res)

sess.close()