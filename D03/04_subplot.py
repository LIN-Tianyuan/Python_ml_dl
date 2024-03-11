"""
Subfigure: Matrix Layout
"""
import matplotlib.pyplot as plt
plt.figure('subplot', facecolor='lightgray')

# Subchart
# for i in range(1, 10):
#     plt.subplot(3, 3, i)
#     plt.plot([1, 2, 3], [1, 2, 3])
#     plt.plot([1, 2, 3], [3, 2, 1])
#     plt.xticks([])
#     plt.yticks([])

for i in range(1, 10):
    plt.subplot(3, 3, i)
    # vertical: top center bottom
    # horizontal: left center right
    plt.text(0.5, 0.5, i,
             fontsize=28,
             ha='center',
             va='center')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()