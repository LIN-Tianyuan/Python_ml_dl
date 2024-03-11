"""
Column-level operations on dataframe: CRUD
"""
import pandas as pd

data = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
        'two': pd.Series([1, 2, 3, 4], index=['a','b','c','d']),
        'three': pd.Series([1, 3, 4], index=['a', 'c', 'd'])}
df = pd.DataFrame(data)
print(df)
"""
   one  two  three
a  1.0    1    1.0
b  2.0    2    NaN
c  3.0    3    3.0
d  NaN    4    4.0
"""

# Column access
# Column level indexes, no positional indexes, only tab indexes
# Access to one column: index, access to multiple columns: masks
print(df['one'])
"""
a    1.0
b    2.0
c    3.0
d    NaN
Name: one, dtype: float64
"""
print(df[['one', 'two']])
"""
   one  two
a  1.0    1
b  2.0    2
c  3.0    3
d  NaN    4
"""
# Not the last column.
print(df.columns[:-1])  # Index(['one', 'two'], dtype='object')
print(df[df.columns[:-1]])
"""
   one  two
a  1.0    1
b  2.0    2
c  3.0    3
d  NaN    4
"""

# Addition of columns
# df[column name] = current column's data
# When the value is a list, the number of elements in the list should be equal to the number of indexes.
df['four'] = [1, 2, 3, 4]
print(df)
"""
   one  two  three  four
a  1.0    1    1.0     1
b  2.0    2    NaN     2
c  3.0    3    3.0     3
d  NaN    4    4.0     4
"""
# When the value is Series, the index of the element in the specified Series is the index of df.
df['five'] = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
print(df)
"""
   one  two  three  four  five
a  1.0    1    1.0     1   1.0
b  2.0    2    NaN     2   2.0
c  3.0    3    3.0     3   3.0
d  NaN    4    4.0     4   NaN
"""

# Deletion of columns
del df['five']
print(df)
"""
   one  two  three  four
a  1.0    1    1.0     1
b  2.0    2    NaN     2
c  3.0    3    3.0     3
d  NaN    4    4.0     4
"""

df.pop('four')
print(df)
"""
   one  two  three
a  1.0    1    1.0
b  2.0    2    NaN
c  3.0    3    3.0
d  NaN    4    4.0
"""

df.drop(['two', 'three'], axis=1, inplace=True)
print(df)
"""
   one
a  1.0
b  2.0
c  3.0
d  NaN
"""

# modifications
df['one'] = [4, 3, 2, 1]
print(df)
"""
   one
a    4
b    3
c    2
d    1
"""