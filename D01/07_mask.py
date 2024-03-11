"""
Masking operations: Boolean masks, indexed masks
"""
import numpy as np
# Boolean masks
ary = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
mask = [True, False, True, False,
        True, False, True, False, True]
print(ary[mask])    # [1 3 5 7 9]

# Find multiples of 3 up to 100
ary = np.arange(1, 101)
mask = ary % 3 == 0
print(ary[mask])
"""
[ 3  6  9 12 15 18 21 24 27 30 33 36 39 42 45 48 51 54 57 60 63 66 69 72
 75 78 81 84 87 90 93 96 99]
"""

# Find numbers from 1 to 100 that are divisible by both 3 and 7.
ary = np.arange(1, 101)
mask = (ary % 3 == 0) & (ary % 7 == 0)
print(ary[mask])        # [21 42 63 84]

# index mask
car = np.array(['BMW', 'Benz', 'Audi', 'BYD', 'Tesla'])
mask = [3, 4, 1, 0, 2]
print(car[mask])        # ['BYD' 'Tesla' 'Benz' 'BMW' 'Audi']
mask = [0, 1, 2]
print(car[mask])        # ['BMW' 'Benz' 'Audi']

import numpy as np
a = np.array([[1 + 1j, 2 + 4j, 3 + 7j],
              [4 + 2j, 5 + 5j, 6 + 8j],
              [7 + 3j, 8 + 6j, 9 + 9j]])
print(a.shape)
print(a.dtype)
print(a.ndim)
print(a.size)
print(a.itemsize)
print(a.nbytes)
print(a.real, a.imag, sep='\n')
print(a.T)
print([elem for elem in a.flat])
b = a.tolist()
print(b)
