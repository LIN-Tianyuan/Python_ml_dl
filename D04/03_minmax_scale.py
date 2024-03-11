"""
Range scaling: narrowing the differences between columns
Set the minimum and maximum values of each column to the same interval: (0, 1)
"""
import numpy as np

raw_sample = np.array([[3.0, -100.0, 2000.0],
                       [0.0, 400.0, 3000.0],
                       [1.0, -400.0, 2000.0]])

mms_sample = raw_sample.copy()

# 1. Subtract the minimum value
# 2. Result after subtraction / Polar deviation
for col in mms_sample.T:
    col_min = col.min()
    col_max = col.max()
    col -= col_min
    col /= (col_max - col_min)

print(mms_sample)
"""
[[1.         0.375      0.        ]
 [0.         1.         1.        ]
 [0.33333333 0.         0.        ]]
"""
print("-" * 40)
# Range scaling based on the API provided by sklearn
import sklearn.preprocessing as sp  # Data preprocessing module
# res = sp.minmax_scale(raw_sample)
# print(res)
"""
[[1.         0.375      0.        ]
 [0.         1.         1.        ]
 [0.33333333 0.         0.        ]]
"""

mms = sp.MinMaxScaler()
# mms.fit(raw_sample)
# res = mms.transform(raw_sample)
res = mms.fit_transform(raw_sample)
print(res)

"""
[[1.         0.375      0.        ]
 [0.         1.         1.        ]
 [0.33333333 0.         0.        ]]
"""
