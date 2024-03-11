"""
Common math metrics in numpy and pandas
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

# average value
print(np.mean(fracture))            # 4.0
print(fracture.mean())              # 4.0
print(np.mean(data, axis = 0))
"""
John Carson          3.000000
Michelle Peterson    3.250000
William Reynolds     3.250000
Jillian Hobart       3.500000
Melissa Jones        2.833333
Alex Roberts         3.700000
Michael Henry        3.166667
dtype: float64
"""
print(np.mean(data, axis=1))
"""
Inception           2.800000
Pulp Fiction        3.714286
Anger Management    2.375000
Fracture            4.000000
Serendipity         2.500000
Jerry Maguire       3.416667
dtype: float64
"""
print(data.mean(axis=0))
print(data.mean(axis=1))
"""
John Carson          3.000000
Michelle Peterson    3.250000
William Reynolds     3.250000
Jillian Hobart       3.500000
Melissa Jones        2.833333
Alex Roberts         3.700000
Michael Henry        3.166667
dtype: float64
Inception           2.800000
Pulp Fiction        3.714286
Anger Management    2.375000
Fracture            4.000000
Serendipity         2.500000
Jerry Maguire       3.416667
dtype: float64
"""