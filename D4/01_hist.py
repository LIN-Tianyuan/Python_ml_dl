"""
bar chart
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(175, 5, 20000)
plt.hist(data, bins=100)
plt.show()
