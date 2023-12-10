"""
Common math metrics for numpy, pandas
"""
import numpy as np
import pandas as pd

data = pd.read_json('../data_test/ratings.json')
print(data)
"""
                  John Carson  Michelle Peterson  ...  Alex Roberts  Michael Henry
Inception                 2.5                3.0  ...           3.0            NaN
Pulp Fiction              3.5                3.5  ...           4.0            4.5
Anger Management          3.0                1.5  ...           NaN            NaN
Fracture                  3.5                5.0  ...           5.0            4.0
Serendipity               2.5                3.5  ...           3.5            1.0
Jerry Maguire             3.0                3.0  ...           3.0            NaN

[6 rows x 7 columns]
"""

fracture = data.loc['Fracture']
print(fracture)
"""
John Carson          3.5
Michelle Peterson    5.0
William Reynolds     3.5
Jillian Hobart       4.0
Melissa Jones        3.0
Alex Roberts         5.0
Michael Henry        4.0
Name: Fracture, dtype: float64
"""

# weighted average
weights = [1, 10, 1, 1, 1, 10, 1]
print(np.average(fracture, weights=weights))    # 4.72

# Maximum value and index of the maximum value
print('{} gave Fracture top marks for this movie:{}'.format(
    np.argmax(fracture), np.max(fracture)))
# 1 gave Fracture top marks for this movie:5.0

print('{} gave Fracture top marks for this movie:{}'.format(
    fracture.idxmax(), fracture.max()))
# Michelle Peterson gave Fracture top marks for this movie:5.0

# median
print(np.median(fracture))  # 4.0

# (statistics) standard deviation
print(fracture.std())               # 0.7637626158259734
print(np.std(fracture))             # 0.7071067811865476
print(np.std(fracture, ddof=1))     # 0.7637626158259734