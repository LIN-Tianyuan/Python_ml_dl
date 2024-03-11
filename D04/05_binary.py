"""
binarization
Set thresholds that will be greater than the threshold -> 1,
less than or equal to the threshold -> 0
"""
import numpy as np

raw_sample = np.array([[77.7, 88.8, 99.9],
                      [55.5, 22.2, 65.4],
                      [12.3, 45.6, 78.9]])
nor_sample = raw_sample.copy()

# Turns passing data to 1 and failing data to 0
# Threshold: 59.9

# Take data less than or equal to 59.9 and assign it to 0.
# nor_sample[nor_sample <= 59.9] = 0
# Take the data greater than 59.9 and assign it to 1.
# nor_sample[nor_sample > 59.9] = 1

# print(nor_sample)
"""
[[1. 1. 1.]
 [0. 0. 1.]
 [0. 0. 1.]]
"""

# np,where(condition, return value if condition holds, return value if it doesn't)
res = np.where(nor_sample>59.5, 1.0, 0.0)
print(res)
"""
[[1. 1. 1.]
 [0. 0. 1.]
 [0. 0. 1.]]
"""
print('-' * 40)
# Implementation of binarization based on the API provided by sklearn
import sklearn.preprocessing as sp  # Data preprocessing module
print(sp.binarize(raw_sample, threshold=59.9))
"""
[[1. 1. 1.]
 [0. 0. 1.]
 [0. 0. 1.]]
"""
print('-' * 40)
# object
biner = sp.Binarizer(threshold=59.9)
print(biner.transform(raw_sample))
"""
[[1. 1. 1.]
 [0. 0. 1.]
 [0. 0. 1.]]
"""