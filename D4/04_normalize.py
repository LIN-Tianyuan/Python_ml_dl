"""
Normalization: converting data into a percentage of each row
"""
import numpy as np

raw_sample = np.array([[10.0, 20.0, 5.0],
                       [8.0, 10.0, 1.0]])

nor_sample = raw_sample.copy()
for row in nor_sample:
    row /= abs(row).sum()

print(nor_sample)
print(nor_sample.sum(axis=1))

"""
[[0.28571429 0.57142857 0.14285714]
 [0.42105263 0.52631579 0.05263158]]
[1. 1.]
"""
print('-' * 40)
# Normalization based on the API provided by sklearn
import sklearn.preprocessing as sp  # Data preprocessing module
res = sp.normalize(raw_sample, norm='l1')
print(res)

"""
[[0.28571429 0.57142857 0.14285714]
 [0.42105263 0.52631579 0.05263158]]

"""