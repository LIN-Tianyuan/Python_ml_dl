"""
Clothing Recognition: Convolutional Neural Networks
"""
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow as tf


class FashionMnist():
    out_feature1 = 12   # Number of convolutional kernels in the first set
    out_feature2 = 24   # Number of convolutional kernels in the second set
    con_neurons = 512   # Number of neurons in the fully connected layer

    def __init__(self, path):
        self.data = read_data_sets(path, one_hot=True)
        self.sess = tf.Session()

    def close(self):
        self.sess.close()

    # Initialization weights
    def init_weight_var(self, shape):
        # truncated normal distribution
        init_w = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init_w)

    # Initialization bias
    def init_bias_var(self, shape):
        init_b = tf.constant(1.0, shape=shape)
        return tf.Variable(init_b)

    # two-dimensional convolution
    def conv2d(self, x, w):
        return tf.nn.conv2d(x,  # input data
                            w,  # convolution kernel (math.)
                            strides=[1, 1, 1, 1],   # footsteps    NHWC
                            padding='SAME')

    # pooling
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x,   # input data
                              ksize=[1, 2, 2, 1],      # Pooling area
                              strides=[1, 2, 2, 1],    # Pooling step
                              padding='SAME')          # padding

    # convolution pooling group
    def create_conv_pool_layer(self, input, input_c, out_c):
        # convolution kernel
        filter_w = self.init_weight_var([5, 5, input_c, out_c])
        # Bias of the convolution kernel
        b_conv = self.init_bias_var([out_c])
        # Perform Convolution, activation
        h_conv = tf.nn.relu(self.conv2d(input, filter_w) + b_conv)
        # implementation pooling
        h_pool = self.max_pool_2x2(h_conv)
        return h_pool

    # full connectivity layer
    def create_fc_layer(self, h_pool_flat, input_feature, con_neurons):
        """
        full connectivity layer
        :param h_pool_flat: Input data (one-dimensional features)
        :param input_feature: Input features
        :param con_neurons: Number of neurons
        :return:
        """
        w_fc = self.init_weight_var([input_feature, con_neurons])
        b_fc = self.init_bias_var([con_neurons])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, w_fc) + b_fc)
        return h_fc1

    def build(self):
        # Forming a CNN
        # Placeholders for sample data
        self.x = tf.placeholder('float32', [None, 784])
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        self.y = tf.placeholder('float32', [None, 10])

        # Group I convolution pooling
        h_pool1 = self.create_conv_pool_layer(x_image, 1, self.out_feature1)

        # Group II Convolutional Pooling
        h_pool2 = self.create_conv_pool_layer(h_pool1, self.out_feature1, self.out_feature2)

        # full connectivity layer
        h_pool2_flat_feature = 7*7*self.out_feature2
        h_pool2_flat = tf.reshape(h_pool2, [-1, h_pool2_flat_feature])
        h_fc = self.create_fc_layer(h_pool2_flat,
                                    h_pool2_flat_feature,
                                    self.con_neurons)
        # dropout layer
        h_fc_drop = tf.nn.dropout(h_fc, 0.5)

        # output layer
        w_fc = self.init_weight_var([self.con_neurons, 10])    # (512,0)
        b_fc = self.init_bias_var([10])
        pred_y = tf.matmul(h_fc_drop, w_fc) + b_fc

        # loss function
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y, logits=pred_y
        )
        # find the mean value
        cross_entropy = tf.reduce_mean(loss)

        # gradient descent: Adaptive Gradient Descent Optimizer
        self.train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

        # precision
        corr = tf.equal(tf.argmax(self.y, 1), tf.argmax(pred_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(corr, 'float32'))

    def train(self):
        # initialization
        self.sess.run(tf.global_variables_initializer())
        # batch size
        batch_size = 100
        print('Start training......')
        for i in range(10): # round
            total_batch = int(self.data.train.num_examples / batch_size)
            # Number of inner control batches
            total_acc = 0.0
            for j in range(total_batch):
                # Getting a batch of data from the training set
                train_x, train_y = self.data.train.next_batch(batch_size)
                params = {self.x: train_x, self.y: train_y}
                t, acc = self.sess.run([self.train_op, self.accuracy], feed_dict=params)

                # Average precision
                total_acc += acc
            avg_acc = total_acc / total_batch
            print('Round:{}, Accuracy:{}'.format(i+1, avg_acc))

    # valuation
    def metrics(self):
        test_x, test_y = self.data.test.next_batch(10000)
        params = {self.x: test_x, self.y: test_y}
        test_acc = self.sess.run(self.accuracy, feed_dict=params)
        print('Test Set Accuracy:', test_acc)


if __name__ == '__main__':
    mnist = FashionMnist('../fashion_mnist/')
    mnist.build()       # network building
    mnist.train()       # train
    mnist.metrics()     # valuation
    mnist.close()

"""
Start training......
Round:1, Accuracy:0.8308181830834259
Round:2, Accuracy:0.9476181834394282
Round:3, Accuracy:0.9638363684307445
Round:4, Accuracy:0.9705090982263739
Round:5, Accuracy:0.9750000088865107
Round:6, Accuracy:0.9789818297732961
Round:7, Accuracy:0.9809636467153375
Round:8, Accuracy:0.9828545558452606
Round:9, Accuracy:0.9845636469667608
Round:10, Accuracy:0.9853091019933874
Test Set Accuracy: 0.9818
"""