"""
Kmeans clustering
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
model = sc.KMeans(n_clusters=4)
# Train
model.fit(data)
# Clustering results
labels = model.labels_
# print(labels)
"""
[2 2 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 1 1 0 3 2 1 0 3 2 1 0 3 2
 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1
 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 0 1 0 3 2 1 0 3 2 1 0 3 2 1 0
 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3
 2 2 0 3 2 1 0 3 2 1 0 1 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2
 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3]
"""
# Geometric center (cluster center)
center = model.cluster_centers_
# print(center)
"""
[[5.91196078 2.04980392]
 [3.1428     5.2616    ]
 [7.07326531 5.61061224]
 [1.831      1.9998    ]]
"""
# Execute prediction
pred_y = model.predict([[1.1, 2.2]])
print(pred_y)   # [1]
# Visualization
plt.scatter(data['x1'], data['x2'],
            c=labels, cmap='brg')
plt.colorbar()
plt.scatter(center[:, 0], center[:, 1],
            color='black',
            marker='+',
            s=300)
plt.show()