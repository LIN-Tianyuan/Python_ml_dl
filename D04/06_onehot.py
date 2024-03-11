"""
Unique thermal code
Constructs a sequence of one 1 and a number of zeros based on the number of non-repeating values in the features
"""
import numpy as np
import sklearn.preprocessing as sp

data = np.array([[1, 3, 2,],
                [7, 5, 4],
                [1, 8, 6],
                [7, 3, 9]])

encoder = sp.OneHotEncoder(sparse=False,
                           dtype='int32')
res = encoder.fit_transform(data)
print(res)
"""
[[1 0 1 0 0 1 0 0 0]
 [0 1 0 1 0 0 1 0 0]
 [1 0 0 0 1 0 0 1 0]
 [0 1 1 0 0 0 0 0 1]]
"""
print(encoder.inverse_transform(res))
"""
[[1 3 2]
 [7 5 4]
 [1 8 6]
 [7 3 9]]
"""