"""
Basic creation and characteristics of nparray arrays
"""
import numpy as np

# np.array
ary = np.array([1, 2, 3, 4, 5, 6])
print(ary)  # [1 2 3 4 5 6]
print(type(ary))    # <class 'numpy.ndarray'>
# Broadcast mechanism:
# when an array is computed with one element, each element in the array is computed separately
print(ary + 3)  # [4 5 6 7 8 9]
print(ary * 2)  # [ 2  4  6  8 10 12]
print(ary == 3) # [False False  True False False False]
# Arrays and arrays are computed at the corresponding positions.
print(ary + ary)    # [ 2  4  6  8 10 12]
print(ary * ary)    # [ 1  4  9 16 25 36]

ary = np.array([[1, 2, 3],
                [4, 5, 6]])
print(ary)

# np.arange
print('-' * 40)
ary = np.arange(1, 11)
print(ary)  # [ 1  2  3  4  5  6  7  8  9 10]

# np.zeros
print('-' * 40)
zeros = np.zeros(10, dtype='int32')
print(zeros)    # [0 0 0 0 0 0 0 0 0 0]
zeros = np.zeros(shape=(2, 3), dtype='int32')
print(zeros)
"""
[[0 0 0]
 [0 0 0]]
"""

# np.ones
print('-' * 40)
ones = np.ones(shape=(2, 3), dtype='int32')
print(ones)
"""
[[1 1 1]
 [1 1 1]]
"""

# np.zeros_like()
print('-' * 40)
zeros_like = np.zeros_like(ary)
print(zeros_like)   # [0 0 0 0 0 0 0 0 0 0]

# np.ones_like()
print('-' * 40)
ones_like = np.ones_like(ary)
print(ones_like)   # [1 1 1 1 1 1 1 1 1 1]

# Generate an array of values all 0.2, shape=(10,)
print('-' * 40)
ary = np.zeros(shape=(10,)) + 0.2
print(ary)  # [0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2]

# Generate 100 numbers in -3.14 to 3.14
# linear splitting
print('-' * 40)
pai = np.linspace(-3.14, 3.14, 100)
print(pai)
"""
[-3.14       -3.07656566 -3.01313131 -2.94969697 -2.88626263 -2.82282828
 -2.75939394 -2.6959596  -2.63252525 -2.56909091 -2.50565657 -2.44222222
 -2.37878788 -2.31535354 -2.25191919 -2.18848485 -2.12505051 -2.06161616
 -1.99818182 -1.93474747 -1.87131313 -1.80787879 -1.74444444 -1.6810101
 -1.61757576 -1.55414141 -1.49070707 -1.42727273 -1.36383838 -1.30040404
 -1.2369697  -1.17353535 -1.11010101 -1.04666667 -0.98323232 -0.91979798
 -0.85636364 -0.79292929 -0.72949495 -0.66606061 -0.60262626 -0.53919192
 -0.47575758 -0.41232323 -0.34888889 -0.28545455 -0.2220202  -0.15858586
 -0.09515152 -0.03171717  0.03171717  0.09515152  0.15858586  0.2220202
  0.28545455  0.34888889  0.41232323  0.47575758  0.53919192  0.60262626
  0.66606061  0.72949495  0.79292929  0.85636364  0.91979798  0.98323232
  1.04666667  1.11010101  1.17353535  1.2369697   1.30040404  1.36383838
  1.42727273  1.49070707  1.55414141  1.61757576  1.6810101   1.74444444
  1.80787879  1.87131313  1.93474747  1.99818182  2.06161616  2.12505051
  2.18848485  2.25191919  2.31535354  2.37878788  2.44222222  2.50565657
  2.56909091  2.63252525  2.6959596   2.75939394  2.82282828  2.88626263
  2.94969697  3.01313131  3.07656566  3.14      ]
"""