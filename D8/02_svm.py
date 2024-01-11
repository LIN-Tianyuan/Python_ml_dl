"""
Support vector machines:
finding the optimal classification hyperplane
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms    # Model Selection
import sklearn.svm as svm  # Support vector machines
import sklearn.metrics as sm    # valuation

data = pd.read_csv('../data_test/multiple2.txt',
                   header=None,
                   names=['x1','x2','y'])
# print(data.head())
"""
     x1    x2  y
0  5.35  4.48  0
1  6.72  5.37  0
2  3.57  5.25  0
3  4.77  7.65  1
4  2.25  4.07  1
"""

# plt.scatter(data['x1'], data['x2'],
#             c=data['y'], cmap='brg')
# plt.colorbar()
# plt.show()

x = data.iloc[:, :-1]
y = data.iloc[:, -1]
# Divide the training set and test set
train_x, test_x, train_y, test_y = ms.train_test_split(x, y,
                                                       test_size=0.2,
                                                       random_state=7,
                                                       stratify=y)
'''
# Build a model
model = svm.SVC(kernel='linear')
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
# print(sm.classification_report(test_y, pred_test_y))
"""
              precision    recall  f1-score   support

           0       0.60      0.93      0.73        30
           1       0.85      0.37      0.51        30

    accuracy                           0.65        60
   macro avg       0.72      0.65      0.62        60
weighted avg       0.72      0.65      0.62        60
"""

# Violent mapping of classification boundary lines
# 1.Split 200 numbers from the minimum value of x1 to the maximum value of x1
x1s = np.linspace(data['x1'].min(), data['x1'].max(), 200)
# 2.Split the minimum value of x2 to the maximum value of x2 by 200 numbers
x2s = np.linspace(data['x2'].min(), data['x2'].max(), 200)
# 3.Combine all cases of x1 and x2, 4w points
points = []
for x1 in x1s:
    for x2 in x2s:
        points.append([x1, x2])
points = pd.DataFrame(points,
                      columns=['x1', 'x2'])
# 4.Substituting 4w points into the model gives the prediction category
points_label = model.predict(points)
# 5.Draw a scatterplot of 4w points, with colors changing according to the prediction category
plt.scatter(points['x1'], points['x2'],
            c=points_label, cmap='gray')
# 6.Plot the scatterplot of the sample
plt.scatter(data['x1'], data['x2'],
            c=data['y'], cmap='brg')
plt.show()

# Build a model
model = svm.SVC(kernel='poly', degree=4)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.classification_report(test_y, pred_test_y))
"""
              precision    recall  f1-score   support

           0       0.86      0.83      0.85        30
           1       0.84      0.87      0.85        30

    accuracy                           0.85        60
   macro avg       0.85      0.85      0.85        60
weighted avg       0.85      0.85      0.85        60
"""

# Violent mapping of classification boundary lines
# 1.Split 200 numbers from the minimum value of x1 to the maximum value of x1
x1s = np.linspace(data['x1'].min(), data['x1'].max(), 200)
# 2.Split the minimum value of x2 to the maximum value of x2 by 200 numbers
x2s = np.linspace(data['x2'].min(), data['x2'].max(), 200)
# 3.Combine all cases of x1 and x2, 4w points
points = []
for x1 in x1s:
    for x2 in x2s:
        points.append([x1, x2])
points = pd.DataFrame(points,
                      columns=['x1', 'x2'])
# 4.Substituting 4w points into the model gives the prediction category
points_label = model.predict(points)
# 5.Draw a scatterplot of 4w points, with colors changing according to the prediction category
plt.scatter(points['x1'], points['x2'],
            c=points_label, cmap='gray')
# 6.Plot the scatterplot of the sample
plt.scatter(data['x1'], data['x2'],
            c=data['y'], cmap='brg')
plt.show()
'''
# Build a model
model = svm.SVC(kernel='rbf', gamma=0.1, C=1.0)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.classification_report(test_y, pred_test_y))
"""
              precision    recall  f1-score   support

           0       0.90      0.93      0.92        30
           1       0.93      0.90      0.92        30

    accuracy                           0.92        60
   macro avg       0.92      0.92      0.92        60
weighted avg       0.92      0.92      0.92        60
"""

# Violent mapping of classification boundary lines
# 1.Split 200 numbers from the minimum value of x1 to the maximum value of x1
x1s = np.linspace(data['x1'].min(), data['x1'].max(), 200)
# 2.Split the minimum value of x2 to the maximum value of x2 by 200 numbers
x2s = np.linspace(data['x2'].min(), data['x2'].max(), 200)
# 3.Combine all cases of x1 and x2, 4w points
points = []
for x1 in x1s:
    for x2 in x2s:
        points.append([x1, x2])
points = pd.DataFrame(points,
                      columns=['x1', 'x2'])
# 4.Substituting 4w points into the model gives the prediction category
points_label = model.predict(points)
# 5.Draw a scatterplot of 4w points, with colors changing according to the prediction category
plt.scatter(points['x1'], points['x2'],
            c=points_label, cmap='gray')
# 6.Plot the scatterplot of the sample
plt.scatter(data['x1'], data['x2'],
            c=data['y'], cmap='brg')
plt.show()