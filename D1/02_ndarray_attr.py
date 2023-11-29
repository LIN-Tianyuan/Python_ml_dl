"""
Test basic properties of data
"""
import numpy as np

# shape
ary = np.arange(1, 11)
print(ary)  # [ 1  2  3  4  5  6  7  8  9 10]
print(ary.shape)    # (10,)

ary.shape = (2, 5)
print(ary)
"""
[[ 1  2  3  4  5]
 [ 6  7  8  9 10]]
"""

ary.shape = (5, 2)
print(ary)
"""
[[ 1  2]
 [ 3  4]
 [ 5  6]
 [ 7  8]
 [ 9 10]]
"""

ary.shape = (1, 2, 5)
print(ary)
"""
[[[ 1  2  3  4  5]
  [ 6  7  8  9 10]]]
"""

# dtype
print('-' * 40)
ary = np.arange(1, 9)
print(ary)  # [1 2 3 4 5 6 7 8]
print(ary.dtype)    # int64

bry = ary.astype('float64')
print(bry)  # [1. 2. 3. 4. 5. 6. 7. 8.]

# size
print('-' * 40)
ary = np.arange(1, 9)
print(ary)  # [1 2 3 4 5 6 7 8]
print(ary.size)  # 8
print(len(ary))  # 8

ary.shape = (2, 4)
print(ary.size)  # 8
print(len(ary))  # 2

# index
print('-' * 40)
ary = np.arange(1, 9)
ary.shape = (2, 2, 2)
print(ary)
"""
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
"""
print(ary[0])
"""
[[1 2]
 [3 4]]
"""
print(ary[0][0])
"""
[1 2]
"""
print(ary[0][0][0])  # 1
# Three-dimensional arrays [index of page, index of row, index of column]
print(ary[0])
"""
[[1 2]
 [3 4]]
"""
print(ary[0, 0])
"""
[1 2]
"""
print(ary[0, 0, 0])  # 1

print('-' * 40)
ary = np.ones(shape=(2, 2),
              dtype='bool_')
print(ary)
"""
[[ True  True]
 [ True  True]]
"""

print('-' * 40)
# 自定义复合类型
import numpy as np

data=[
	('zs', [90, 80, 85], 15),
	('ls', [92, 81, 83], 16),
	('ww', [95, 85, 95], 15)
]
#第一种设置dtype的方式
a = np.array(data, dtype='U3, 3int32, int32')
print(a)
print(a[0]['f0'], ":", a[1]['f1'])
print("=====================================")

#第二种设置dtype的方式
c = np.array(data, dtype={'names': ['name', 'scores', 'ages'],
                    'formats': ['U3', '3int32', 'int32']})
print(c[0]['name'], ":", c[0]['scores'], ":", c.itemsize)
print("=====================================")

#测试日期类型数组
f = np.array(['2011', '2012-01-01', '2013-01-01 01:01:01','2011-02-01'])
f = f.astype('M8[D]')
f = f.astype('i4')
print(f[3]-f[0])

f.astype('bool')