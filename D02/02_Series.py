"""
Series Example: Creation and Access
"""
import pandas as pd
import numpy as np
s = pd.Series([100, 98, 66, 23])
print(s)
"""
0    100
1     98
2     66
3     23
dtype: int64
"""
s = pd.Series([100, 98, 66, 23],
              index=['zs', 'ls', 'ww', 'sl'])
print(s)
"""
zs    100
ls     98
ww     66
sl     23
dtype: int64
"""
data = {'zs': 100, 'ls': 98, 'ww': 66, 'sl': 23}
s = pd.Series(data)
print(s)
"""
zs    100
ls     98
ww     66
sl     23
dtype: int64
"""
s = pd.Series(5,
              index=np.arange(10))
print(s)
"""
0    5
1    5
2    5
3    5
4    5
5    5
6    5
7    5
8    5
9    5
dtype: int64
"""

# Accessing data in a Series: indexes, slices, masks
print('-' * 40)
# There are two sets of indexes in Series: the location index and the label index.
s = pd.Series([100, 90, 80, 70],
              index=['zs', 'ls', 'ww', 'sl'])
# Location indexing: index, slice, mask
# There are only forward indexes in Series, not reverse indexes.
print(s[3])     # 70
print(s[-1])    # 70  s['sl']
print(s[:2])
"""
zs    100
ls     90
dtype: int64
"""
print(s[[0, 1, 3]])
"""
zs    100
ls     90
sl     70
dtype: int64
"""

# Tag indexing: indexing, slicing, masking
print(s['sl'])  # 70
print(s['zs':'ww']) # Label index slice contains the termination position
"""
zs    100
ls     90
ww     80
dtype: int64
"""
print(s[['zs','ls','sl']])
"""
zs    100
ls     90
sl     70
dtype: int64
"""
print(s.ndim)       # 1
print(s.shape)      # (4,)
print(s.dtype)      # int64
print(s.size)       # 4
print(s.values)     # [100  90  80  70]
print(s.index)      # Index(['zs', 'ls', 'ww', 'sl'], dtype='object')
