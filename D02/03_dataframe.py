"""
dataframe example: dataframe creation
"""
import pandas as pd

# one-dimensional list
data = [100, 90, 80, 70]
df = pd.DataFrame(data)
print(df)
"""
     0
0  100
1   90
2   80
3   70
"""

# two-dimensional list
data = [['Tom', 18],
        ['Jerry', 18],
        ['Jack', 20],
        ['Rose', 20]]
df = pd.DataFrame(data,
                  index=['s01', 's02', 's03', 's04'],
                  columns=['Name', 'Age'])
print(df)
"""
      Name  Age
s01    Tom   18
s02  Jerry   18
s03   Jack   20
s04   Rose   20
"""

# Convert dictionary to df
# If the value is a list, the number of elements in each list should be the same.
data = {'Name': ['Tom', 'Jerry', 'Jack', 'Rose'],
        'Age': [18, 18, 20, 20]}
df = pd.DataFrame(data)
print(df)
"""
    Name  Age
0    Tom   18
1  Jerry   18
2   Jack   20
3   Rose   20
"""

data = {'Name': pd.Series(['Tom', 'Jerry', 'Jack', 'Rose']),
        'Age': pd.Series([18, 18, 20])}
df = pd.DataFrame(data)
print(df)
"""
    Name   Age
0    Tom  18.0
1  Jerry  18.0
2   Jack  20.0
3   Rose   NaN
"""

data = {'Name': pd.Series(['Tom', 'Jerry', 'Jack', 'Rose'],
                          index=['s01', 's02', 's03', 's04']),
        'Age': pd.Series([18, 18, 20], index=['s01', 's02', 's04'])}
df = pd.DataFrame(data)
print(df)
"""
      Name   Age
s01    Tom  18.0
s02  Jerry  18.0
s03   Jack   NaN
s04   Rose  20.0
"""

print(df.axes)      # [Index(['s01', 's02', 's03', 's04'], dtype='object'), Index(['Name', 'Age'], dtype='object')]
print(df.index)     # Index(['s01', 's02', 's03', 's04'], dtype='object')
print(df.columns)   # Index(['Name', 'Age'], dtype='object')
print(df.empty)     # False
print(df.ndim)      # 2
print(df.shape)     # (4, 2)
print(df.size)      # 8
print(df.dtypes)
"""
Name     object
Age     float64
dtype: object
"""
print(df.values)
"""
[['Tom' 18.0]
 ['Jerry' 18.0]
 ['Jack' nan]
 ['Rose' 20.0]]
"""
print(df.head(2))
"""
      Name   Age
s01    Tom  18.0
s02  Jerry  18.0
"""
print(df.tail(2))
"""
     Name   Age
s03  Jack   NaN
s04  Rose  20.0
"""