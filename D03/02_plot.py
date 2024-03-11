"""
matplotlib basic drawing
"""
import numpy as np
import matplotlib.pyplot as plt

# 绘制简单直线
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 6, 9, 12, 15])

# plt.plot(x, y)

# Draw the image of the sine function from -π to +π   np.linspace
xs = np.linspace(-np.pi, np.pi, 200)
# sin
sinx = np.sin(xs)
plt.plot(xs, sinx,
		 linestyle='--',
		 linewidth=3,
		 color='dodgerblue',
         label=r'$y=sin(x)$')
# cos
cosx = np.cos(xs) / 2
plt.plot(xs, cosx,
		 linestyle='-.',
		 linewidth=5,
		 color='orangered',
		 alpha=0.5,
         label=r'$y=\frac{1}{2}cos(x)$')


# Setting the axis display range (first quadrant)
# plt.xlim(0, np.pi + 0.1)
# plt.ylim(0, 1 + 0.1)

# Setting the Axis Scale
plt.xticks([-np.pi, -np.pi / 2.0, 0,
			np.pi / 2, np.pi],
		   [r'$-\pi$', r'$-\frac{\pi}{2}$',
			r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'],
		   fontsize=14)

# Setting the axes
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))

plt.xticks(fontsize=14)
plt.yticks([-1, 0.5, 0.5, 1],
           fontsize=14)

# legend
plt.legend()

# special point
plt.scatter([np.pi/2, -np.pi/2],
            [1, -1],
            marker='*',
            s=300,
            edgecolors='red',
            facecolor='green',
            zorder=2)

plt.show() # Display image, blocking method

