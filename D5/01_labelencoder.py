"""
Tag encoding: assign string -> numeric value
Assigns a numeric value to a string based on the position of the feature value in the sequence of features
"""
import numpy as np
import sklearn.preprocessing as sp  # Data preprocessing module

# Converting one-dimensional data
data = np.array(['benz', 'audi', 'benz', 'bmw',
                 'BYD', 'Tesla', 'bmw', 'bmw'])
encoder = sp.LabelEncoder()
res = encoder.fit_transform(data)
print(res)
"""
[3 2 3 4 0 1 4 4]
"""

# Converting 2D data
data = np.array([['benz', 'c63'],
                 ['audi', 'rs4'],
                 ['bmw', 'M3'],
                 ['BYD', 'U8']])
result = []
encoders = []
for col in data.T:
    encoder = sp.LabelEncoder()
    res = encoder.fit_transform(col)
    result.append(res)
    encoders.append(encoder)

result = np.array(result).T
print(result)

# reverse transitions
print(encoders)
for col in range(len(result.T)):
    encoder = encoders[col]
    res = encoder.inverse_transform(result.T[col])
    print(res)
"""
['benz' 'audi' 'bmw' 'BYD']
['c63' 'rs4' 'M3' 'U8']
"""