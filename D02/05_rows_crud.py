"""
Row-level operations on dataframes: CRUD
"""
import pandas as pd

data = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
        'two': pd.Series([1, 2, 3, 4], index=['a','b','c','d']),
        'three': pd.Series([1, 3, 4], index=['a', 'c', 'd'])}
df = pd.DataFrame(data)
print(df)

# Row access: Cannot be indexed directly, but can be sliced directly
print(df[0:1])
"""
   one  two  three
a  1.0    1    1.0
"""
# df.loc[]  df.iloc[] Think of it directly as a two-dimensional array
# df.loc [operations on rows, operations on columns]
print(df.loc['a'])
"""
one      1.0
two      1.0
three    1.0
Name: a, dtype: float64
"""
print(df.iloc[0])
"""
one      1.0
two      1.0
three    1.0
Name: a, dtype: float64
"""

# All rows, not last column (2D)
print(df.iloc[:, :-1])
"""
   one  two
a  1.0    1
b  2.0    2
c  3.0    3
d  NaN    4
"""
# All rows, as long as the last column (one-dimensional)
print(df.iloc[:, -1])
"""
a    1.0
b    NaN
c    3.0
d    4.0
Name: three, dtype: float64
"""

# Addition of rows  df.append(df2)
df = pd.DataFrame([['Tom', 18],
                  ['Jerry', 22]],
                  columns=['Name', 'Age'])
df2 = pd.DataFrame([['Jack', 18],
                    ['Rose', 21]],
                   columns=['Name', 'Age'])
df = df.append(df2)
df.index = [0, 1, 2, 3]
print(df)
"""
    Name  Age
0    Tom   18
1  Jerry   22
2   Jack   18
3   Rose   21
"""

# Deletion of rows
df.drop([1, 2], axis=0, inplace=True)
print(df)
"""
   Name  Age
0   Tom   18
3  Rose   21
"""

# modifications
df.loc[3] = ['John', 99]
print(df)
"""
   Name  Age
0   Tom   18
3  John   99
"""

# Find rows by columns and find this element for assignment
print(df['Age'][0])  # 18
# Find columns by rows to find this element for assignment
df.loc[0]['Age'] = 666
print(df)
"""
   Name  Age
0   Tom   18
3  John   99
"""