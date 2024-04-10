"""
Handwriting Recognition
Model: Fully Connected Model
"""
import os.path

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Data preparation
mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)
print(mnist.train.next_batch(1)[0].shape)   # (1, 784)

# placeholder
x = tf.placeholder('float32', [None, 784])
y = tf.placeholder('float32', [None, 10])
# model parameter
init_w = tf.random_normal([784, 10])
w = tf.Variable(init_w)
init_b = tf.zeros([10])
b = tf.Variable(init_b)
# Build a model
pred_y = tf.nn.softmax(tf.matmul(x, w) + b)
# Loss function: cross entropy
cross_entropy = -tf.reduce_sum(y * tf.log(pred_y), reduction_indices=1)  # horizontal direction
cost = tf.reduce_mean(cross_entropy)
# gradient descent
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

batch_size = 100
saver = tf.train.Saver()

# Execution
# with tf.Session() as sess:
#     # initialization
#     sess.run(tf.global_variables_initializer())
#
#     # Check to see if the model is saved before training, and if so load it
#     if os.path.exists('../model/mnist/checkpoint'):
#         saver.restore(sess, '../model/mnist/')
#     # Executive training
#     for epoch in range(10):
#         total_batch = int(mnist.train.num_examples / batch_size)
#         total_cost = 0.0
#         for i in range(total_batch):
#             # Getting a batch of data from the training set
#             train_x, train_y = mnist.train.next_batch(batch_size)
#             params = {x: train_x, y: train_y}
#             o, c = sess.run([train_op, cost], feed_dict=params)
#             total_cost += c
#         avg_cost = total_cost / total_batch
#         print('Round:{}, Loss value:{}'.format(epoch, avg_cost))
#     print('End of training')
#
#     # valuation: accuracy
#     corr = tf.equal(tf.argmax(y, 1), tf.argmax(pred_y, 1))
#     tf.cast(corr, 'float32')
#     accuracy = tf.reduce_mean(tf.cast(corr, 'float32'))
#     acc = sess.run(accuracy,
#                    feed_dict={x:mnist.test.images,
#                               y:mnist.test.labels})
#     print('Accuracy is: ', acc)
#
#     # Save Model
#     saver.save(sess, '../model/mnist/')
#     print('Model saved successfully.')

"""
Round:0, Loss value:0.9473477792739868
Round:1, Loss value:0.9276920986717397
Round:2, Loss value:0.9097460368546573
Round:3, Loss value:0.8930410126664422
Round:4, Loss value:0.8774419302831996
Round:5, Loss value:0.8628843608498573
Round:6, Loss value:0.8493167263269424
Round:7, Loss value:0.8362994652986526
Round:8, Loss value:0.8243620225245303
Round:9, Loss value:0.812896747155623
End of training
Accuracy is:  0.8409
Model saved successfully.
"""

# Load the model and perform the prediction
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Load the model
    saver.restore(sess, '../model/mnist/')

    # Read 2 random images from the test set
    test_x, test_y = mnist.test.next_batch(2)
    pred_test_y = sess.run(pred_y,
                           feed_dict={x: test_x})

    # Forecast category
    pred_v = tf.argmax(pred_test_y, 1)
    # Real category
    y_true = tf.argmax(test_y, 1)
    # Predictive probability
    output = tf.reduce_max(pred_test_y, 1)

    print('Real category: ', y_true.eval())
    print('Forecast category: ', pred_v.eval())
    print('Predictive probability: ', output.eval())
    """
    Real category:  [1 7]
    Forecast category:  [1 7]
    Predictive probability:  [0.91494524 0.9999083 ]
    """

    # Show picture
    import pylab
    img1 = test_x[0].reshape(28, 28)
    pylab.imshow(img1)
    pylab.show()

    img2 = test_x[1].reshape(28, 28)
    pylab.imshow(img2)
    pylab.show()