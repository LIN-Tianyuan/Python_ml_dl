"""
Dimensional (shape) operations on arrays
"""
import numpy as np

# View to dimension: does not modify the dimensions of the original data, data sharing
ary = np.arange(1, 10)
print(ary)  # [1 2 3 4 5 6 7 8 9]

bry = ary.reshape(3, 3)
print(bry)
"""
[[1 2 3]
 [4 5 6]
 [7 8 9]]
"""

# Modifying elements in the ary
ary[0] = 666
print(bry)
"""
[[666   2   3]
 [  4   5   6]
 [  7   8   9]]
"""

ary = np.arange(1, 10).reshape(3, 3)
print(ary)
"""
[[1 2 3]
 [4 5 6]
 [7 8 9]]
"""

print(ary.ravel())  # [1 2 3 4 5 6 7 8 9]

# Duplication of variable dimensions: does not modify the dimensions of the original data, data independence
print('-' * 40)
ary = np.arange(1, 9).reshape(2, 2, 2)
print(ary)
"""
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
"""

bry = ary.flatten()
print(bry)  # [1 2 3 4 5 6 7 8]
ary[0, 0, 0] = 666
print(bry)  # [1 2 3 4 5 6 7 8]

# In-place dimensioning: directly modifying the dimensions of the original data
print('-' * 40)
ary = np.arange(1, 19)
ary.resize(2, 3, 3)
print(ary)
"""
[[[ 1  2  3]
  [ 4  5  6]
  [ 7  8  9]]

 [[10 11 12]
  [13 14 15]
  [16 17 18]]]
"""
