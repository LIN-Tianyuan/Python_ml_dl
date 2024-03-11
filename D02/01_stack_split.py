"""
Combining and Splitting Multidimensional Arrays
"""
import numpy as np

x = np.arange(1, 7).reshape(2, 3)
y = np.arange(7, 13).reshape(2, 3)

# Vertical:v
res = np.vstack((x, y))
print(res)
"""
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
"""
x, y = np.vsplit(res, 2)
print(x)
"""
[[1 2 3]
 [4 5 6]]
"""
print(y)
"""
[[ 7  8  9]
 [10 11 12]]
"""

# Horizontal: h
res = np.hstack((x, y))
print(res)
"""
[[ 1  2  3  7  8  9]
 [ 4  5  6 10 11 12]]
"""
x, y = np.hsplit(res, 2)
print(x)
"""
[[1 2 3]
 [4 5 6]]
"""
print(y)
"""
[[ 7  8  9]
 [10 11 12]]
"""

# Depth: d
res = np.dstack((x, y))
print(res)
"""
[[[ 1  7]
  [ 2  8]
  [ 3  9]]

 [[ 4 10]
  [ 5 11]
  [ 6 12]]]
"""
x, y = np.dsplit(res, 2)
print(x)
"""
[[[1]
  [2]
  [3]]

 [[4]
  [5]
  [6]]]
"""
print(y)
"""
[[[ 7]
  [ 8]
  [ 9]]

 [[10]
  [11]
  [12]]]
"""

print("*" * 40)
import numpy as np
a = np.arange(1, 7).reshape(2, 3)
b = np.arange(7, 13).reshape(2, 3)
# 垂直方向完成组合操作，生成新数组
c = np.vstack((a, b))
res=np.concatenate((a, b), axis=0)
# 通过给出的数组与要拆分的份数，按照某个方向进行拆分，axis的取值同上
print(res)
result = np.split(c, 2, axis=0)
print(result)