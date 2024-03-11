"""
Mean Removal:
changes the mean of each column to 0 and the standard deviation to 1.
Reduces the difference between columns.
"""
import numpy as np

raw_sample = np.array([[3.0, -100.0, 2000.0],
                      [0.0, 400.0, 3000.0],
                      [1.0, -400.0, 2000.0]])

std_sample = raw_sample.copy()

# 1. Subtract the average of the current column
# 2. Deviation / Standard deviation of the current column
for col in std_sample.T:
    col_mean = col.mean()
    col_std = col.std()
    col -= col_mean  # col = col - mean cannot be modified
    col /= col_std

print(std_sample)
print(std_sample.mean(axis=0))
print(std_sample.std(axis=0))

"""
[[ 1.33630621 -0.20203051 -0.70710678]
 [-1.06904497  1.31319831  1.41421356]
 [-0.26726124 -1.1111678  -0.70710678]]
[ 5.55111512e-17  0.00000000e+00 -2.96059473e-16]
[1. 1. 1.]
"""
print("-" * 40)
# Implement mean removal based on the API provided by sklearn
import sklearn.preprocessing as sp  # Data preprocessing module
res = sp.scale(raw_sample)
print(res)
print(res.mean(axis=0))
print(res.std(axis=0))

"""
[[ 1.33630621 -0.20203051 -0.70710678]
 [-1.06904497  1.31319831  1.41421356]
 [-0.26726124 -1.1111678  -0.70710678]]
[ 5.55111512e-17  0.00000000e+00 -2.96059473e-16]
[1. 1. 1.]
"""