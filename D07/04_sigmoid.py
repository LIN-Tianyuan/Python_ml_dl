"""
Fake a set of predicted values, draw sigmoid
"""
import numpy as np
import matplotlib.pyplot as plt

ys = np.linspace(-10, 10, 100)

# res = 1 / (1 + np.e ** -ys)
res = 1 / (1 + np.exp(-ys))

plt.plot(ys, res)
plt.show()