"""
DBSCAN clustering
"""
import pandas as pd
import sklearn.cluster as sc  # clustering
import matplotlib.pyplot as plt

data = pd.read_csv('../data_test/multiple3.txt',
                   header=None,
                   names=['x1', 'x2'])
# print(data.head())
"""
     x1    x2
0  1.96 -0.09
1  2.84  3.16
2  4.74  1.84
3  6.36  4.89
4  1.77  1.55
"""

# Build a model
model = sc.DBSCAN(eps=0.6,
                  min_samples=5)
# Train
model.fit(data)
# Clustering results
labels = model.labels_
print(labels)
"""
[-1  1  0  0  1  2  0  0  1  2  0  0  1  2  0  0  1  2  0 -1  1  2  0  0
 -1  2 -1  0  1  2  0  0  1  2  0  0  1  2  0  0  1  2  0  0  1  2 -1  0
  1  2  0  0  1  2  0  0  1  2  0  0  1  2  0  0  1  2  0  0  1  2  0  0
  1  2  0  0  1  2  0  0  1  2  0  0  1  2  0  0  1  2  0  0  1  2  0  0
 -1  2  0  0  1  2 -1  0  1  2  0 -1  1  2  0  0  1  2 -1  0 -1  2  0  0
  1  2  0  0  1  2 -1  0  1  2  0 -1  1  2  0  0 -1  2  0  0  1  2  0 -1
  1  2  0  0 -1  1  0  0  1  2  0  0  1  2 -1 -1  1  2  0  0 -1  2  0  0
  1 -1  0  0  1  2  0  0  1  2  0  0  1  2  0  0  1  2  0  0  1  2  0  0
  1  2 -1  0  1  2  0  0]
"""

# Visualization
plt.scatter(data['x1'], data['x2'],
            c=labels, cmap='brg')
plt.colorbar()
plt.show()