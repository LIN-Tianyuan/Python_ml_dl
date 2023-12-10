"""
scatterplot
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

height = np.random.normal(175, 5, 2000)
weight = np.random.normal(70, 5, 2000)

plt.scatter(height, weight,
            c=height,
            cmap='jet')
plt.colorbar()
plt.show()

"""
Read the data in Salary_data.csv and draw a scatterplot with YearsExperience as x and Salary as y. 
The color changes as Salary changes to see the distribution of this set of data
"""

data = pd.read_csv('../data_test/Salary_Data.csv')
x = data['YearsExperience']
y = data['Salary']

plt.scatter(x, y, c=y, cmap='jet')
plt.show()