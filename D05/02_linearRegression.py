"""
Python code based implementation of gradient descent to find model parameters
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
y = np.array([5.0, 5.5, 6.0, 6.8, 7.1])

# plt.scatter(x, y)
# plt.show()

# objective function: y = w1 * x + w0
w1 = 1  # Weights, random numbers but not 0
w0 = 1  # Bias, 0 or 1
learning_rate = 0.01    # Learning rate, not too big, not too small
epoch = 300  # rounds

w0s, w1s, losses, epochs = [], [], [], []
# Open loop, update parameters
for i in range(epoch):
    loss = ((w1 * x + w0 - y) ** 2).sum() / 2
    # Print w0 and w1 and loss
    print('round:{:3},w1:{:.8f},w0:{:.8f},loss:{:.8f}'.format(i, w1, w0, loss))

    # Collect model parameters and loss values for visualization
    w0s.append(w0)
    w1s.append(w1)
    losses.append(loss)
    epochs.append(i)
    d0 = (w0 + w1 * x - y).sum()
    d1 = (x * (w1 * x + w0 - y)).sum()
    # update parameters
    w0 = w0 - learning_rate * d0
    w1 = w1 - learning_rate * d1

# print(f'w0: {w0}, w1:{w1}')  # w0: 3.7330883882570403, w1:2.645728402613174

pred_y = w1 * x + w0
# plt.plot(x, pred_y, color='orangered')
# plt.scatter(x, y)
# plt.show()

# Visualization of model parameters and loss values
plt.subplot(3, 1, 1)
plt.plot(epochs, w0s, color='dodgerblue', label='w0')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(epochs, w1s, color='dodgerblue', label='w1')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(epochs, losses, color='orangered', label='losses')
plt.legend()

plt.show()