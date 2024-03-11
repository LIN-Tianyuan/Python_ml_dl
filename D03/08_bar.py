"""
histogram
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate Apple sales for January-December
x = np.arange(1, 13)
apples = np.random.normal(30000, 2000, 12)
plt.bar(x-0.2, apples, 0.4)
plt.xticks(x)

# Orange
oranges = np.random.normal(30000, 2000, 12)
plt.bar(x+0.2, oranges, 0.4)
"""
for i in range(len(x)):
    plt.text(x[i], apples[i], int(apples[i]),
         ha='center', va='bottom')
"""


plt.show()