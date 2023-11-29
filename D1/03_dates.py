"""
Testing time date types in numpy
"""
import numpy as np

data = np.array(['2021',
                 '2022-01-01',
                 '2023-01-01 08:08:08'])
print(data)  # ['2021' '2022-01-01' '2023-01-01 08:08:08']
print(data.dtype)  # <U19

# str --> datetime64
dates1 = data.astype('datetime64')
print(dates1)    # ['2021-01-01T00:00:00' '2022-01-01T00:00:00' '2023-01-01T08:08:08']

dates2 = data.astype('datetime64[D]')
print(dates2)    # ['2021-01-01' '2022-01-01' '2023-01-01']

# datetime64 --> int64
res = dates1.astype('int64')
print(res)  # [1609459200 1640995200 1672560488]

res = dates2.astype('int64')
print(res)  # [18628 18993 19358]