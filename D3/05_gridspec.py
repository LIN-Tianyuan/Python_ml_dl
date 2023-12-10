"""
Subfigure: Grid-format layout
"""
import matplotlib.pyplot as plt

plt.figure('GS', facecolor='lightgray')

# Split grid objects with 3 rows and 3 columns
gs = plt.GridSpec(3, 3)
plt.subplot(gs[0, :2])
plt.text(0.5, 0.5, 'Python_base',
         ha='center', va='center',
         fontsize=18)
plt.xticks([])
plt.yticks([])

plt.subplot(gs[:2, -1])
plt.text(0.5, 0.5, 'Socket',
         ha='center', va='center',
         fontsize=18)
plt.xticks([])
plt.yticks([])

plt.subplot(gs[-1, -2:])
plt.text(0.5, 0.5, 'Django',
         ha='center', va='center',
         fontsize=18)
plt.xticks([])
plt.yticks([])

plt.subplot(gs[-2: , 0])
plt.text(0.5, 0.5, 'AI',
         ha='center', va='center',
         fontsize=18)
plt.xticks([])
plt.yticks([])

plt.subplot(gs[1, 1])
plt.text(0.5, 0.5, 'AID',
         ha='center', va='center',
         fontsize=18)
plt.xticks([])
plt.yticks([])
plt.show()