"""
linear regression (math.)
"""
import os.path

import tensorflow as tf

# Data preparation
x = tf.random_normal([100, 1], mean=1.75, stddev=0.5)
# y = 2x + 5
y = tf.matmul(x, [[2.0]]) + 5.0

# Building a linear model
init_w = tf.random_normal(shape=[1, 1])
weight = tf.Variable(init_w, trainable=True)

bias = tf.Variable(0.0, trainable=True)
pred_y = tf.matmul(x, weight) + bias
# Loss function
loss = tf.reduce_mean(tf.square(y - pred_y))
# gradient descent optimizer
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# Define the collection loss function
tf.summary.scalar('losses', loss)
merged = tf.summary.merge_all()

# Model Saving Objects
saver = tf.train.Saver()
# Execution
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Print Initial Value
    print('weight:{},bias:{}'.format(weight.eval(), bias.eval()))

    fw = tf.summary.FileWriter('../summary/',
                               graph=sess.graph)
    # Check to see if the model is saved, and if so, load it
    if os.path.exists('../model/lr/checkpoint'):
        saver.restore(sess, '../model/lr/')
    # cycle training
    for i in range(100):
        sess.run(train_op)
        # Perform one gradient descent and collect one loss value
        summary = sess.run(merged)
        # Write once to the event file
        fw.add_summary(summary, i)
        print('Round:{}, w:{}, b:{}'.format(i+1, weight.eval(), bias.eval()))
    # End of training, save model
    saver.save(sess, '../model/lr/')