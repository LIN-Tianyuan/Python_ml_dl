"""
tensorboard
"""
import tensorflow as tf

x = tf.constant(100.0, name='xxx')
y = tf.constant(200.0, name='yyy')
add = tf.add(x, y, name='addaddadd')

with tf.Session() as sess:
    print(sess.run(add))
    tf.summary.FileWriter('../summary/',
                          graph=sess.graph)
