"""
Use of variables: for model parameters w,b
Variables must be initialized for use
"""
import tensorflow as tf

init_w = tf.random_normal(shape=(3, 4))
w = tf.Variable(initial_value=init_w)
init_b = tf.zeros(shape=(4, ))
b = tf.Variable(initial_value=init_b)

with tf.Session() as sess:
    # Execute the initialization op, Only session.run can be used
    sess.run(tf.global_variables_initializer())
    w, b = sess.run([w, b])
    print(w)
    print(b)