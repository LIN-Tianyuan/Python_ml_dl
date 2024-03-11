"""
Slicing of arrays
"""
import numpy as np

ary = np.arange(1, 19).reshape(2, 9)
print(ary)
"""
[[ 1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18]]
"""

# Two-dimensional arrays [operations on rows, operations on columns]
# First two columns of row 0
print(ary[:1, :2])  # [[1 2]]
print(ary[0, :2])   # [1 2]

# Column 0 of row 0
print(ary[:1, :1])  # [[1]]
print(ary[0, :1])   # [1]
print(ary[0, 0])    # 1

ary = np.arange(1, 51).reshape(5, 10)
print(ary)
"""
[[ 1  2  3  4  5  6  7  8  9 10]
 [11 12 13 14 15 16 17 18 19 20]
 [21 22 23 24 25 26 27 28 29 30]
 [31 32 33 34 35 36 37 38 39 40]
 [41 42 43 44 45 46 47 48 49 50]]
"""
# Get all the rows, not the last column.
print(ary[:, :-1])
"""
[[ 1  2  3  4  5  6  7  8  9]
 [11 12 13 14 15 16 17 18 19]
 [21 22 23 24 25 26 27 28 29]
 [31 32 33 34 35 36 37 38 39]
 [41 42 43 44 45 46 47 48 49]]
"""
# Get all the rows, just the last column.
print(ary[:, -1])   # [10 20 30 40 50]

import numpy as np
a = np.arange(1, 28)
a.resize(3,3,3)
print(a)
#切出1页
print(a[1, :, :])
#切出所有页的1行
print(a[:, 1, :])
#切出0页的1行1列
print(a[0, :, 1])