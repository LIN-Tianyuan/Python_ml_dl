"""
Custom compound type
Columns can be of different types, but the types must be the same within the same column
"""
import numpy as np
data = [('zs', [100, 100, 100], 18),
        ('ls', [90, 90, 90], 19),
        ('ww', [80, 80, 80], 20)]

ary = np.array(data,
               dtype='U2,3int32,int32')
print(ary)
"""
[('zs', [100, 100, 100], 18) ('ls', [ 90,  90,  90], 19)
 ('ww', [ 80,  80,  80], 20)]
"""
# Find the average age of the current 3 people
print(ary['f2'])    # [18 19 20]
print(ary['f2'].mean())  # 19.0

print('-' * 40)

ary = np.array(data,
               dtype={'names': ['name', 'score', 'age'],
                      'formats': ['U2','3int32','int32']})
print(ary['age'])   # [18 19 20]