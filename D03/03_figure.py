"""
Parameters of the test window
"""
import matplotlib.pyplot as plt

plt.figure('666',
           # figsize=(4, 3),
           facecolor='lightgray')

plt.plot([1, 2, 3], [1, 2, 3])
# Setting some parameters of the chart
plt.title('my title', fontsize=24)
plt.xlabel('xxx', fontsize=18)
plt.ylabel('yyy', fontsize=18)

plt.grid(linestyle=':')
plt.tight_layout()
plt.show()


