"""
Iris category forecast
"""
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as sd  # data set
import sklearn.linear_model as lm   # linear model
import sklearn.model_selection as ms  # Model Selection
import sklearn.metrics as sm    # Evaluation module

'''
iris = sd.load_iris()
# print(iris.keys())
"""
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
"""
# print(iris.DESCR)   # 150 samples, 4 features, 3 categories
# print(iris.feature_names)   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# print(iris.target_names)    # ['setosa' 'versicolor' 'virginica']
# print(iris.data.shape)  # (150, 4)
# print(iris.target)
"""
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
"""

# Integrate inputs and outputs into a dataframe
data = pd.DataFrame(iris.data,
                    columns=iris.feature_names)
data['target'] = iris.target
# print(data)

# Visualization of sepals
plt.scatter(data['sepal length (cm)'],
            data['sepal width (cm)'],
            c=data['target'],
            cmap='brg')
plt.colorbar()

# Visualization of petals
plt.figure()
plt.scatter(data['petal length (cm)'],
            data['petal width (cm)'],
            c=data['target'],
            cmap='brg')
plt.colorbar()

# plt.show()

# Pick two categories out of the total data(1, 2) and do a binary classification
# sub_data = data.iloc[50:]
# sub_data = data.tail(100)
sub_data = data[data['target'] != 0]
# print(sub_data)

# Organize inputs and outputs
x = sub_data.iloc[:, :-1]
y = sub_data.iloc[:, -1]
# Divide the training set and test set
train_x ,test_x, train_y, test_y = ms.train_test_split(x, y,
                                                       test_size=0.1,
                                                       random_state=7)

# Build model
model = lm.LogisticRegression(solver='liblinear')
# training model
model.fit(train_x, train_y)
# anticipate
pred_test_y = model.predict(test_x)
# assessment(accuracy)
# print('Real category: ', test_y.values)
# print('Forecast category: ', pred_test_y)
"""
Real category:  [1 1 2 2 1 1 2 2 2 1]
Forecast category:  [1 1 2 2 1 1 2 2 2 2]
"""
# print((test_y == pred_test_y).mean()) # 0.9
print(sm.accuracy_score(test_y, pred_test_y))   # 0.9
'''

'''
iris = sd.load_iris()
# print(iris.keys())
"""
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
"""
# print(iris.DESCR)   # 150 samples, 4 features, 3 categories
# print(iris.feature_names)   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# print(iris.target_names)    # ['setosa' 'versicolor' 'virginica']
# print(iris.data.shape)  # (150, 4)
# print(iris.target)
"""
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
"""

# Integrate inputs and outputs into a dataframe
data = pd.DataFrame(iris.data,
                    columns=iris.feature_names)
data['target'] = iris.target
# print(data)

# Organize inputs and outputs
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
# Divide the training set and test set
train_x ,test_x, train_y, test_y = ms.train_test_split(x, y,
                                                       test_size=0.1,
                                                       random_state=7)

# Build model
model = lm.LogisticRegression(solver='liblinear')
# training model
model.fit(train_x, train_y)
# anticipate
pred_test_y = model.predict(test_x)
# assessment(accuracy)
print('Real category: ', test_y.values)
print('Forecast category: ', pred_test_y)
"""
Real category:  [2 1 0 1 2 0 1 1 0 1 1 1 0 2 0]
Forecast category:  [2 2 0 2 2 0 1 1 0 1 2 2 0 2 0]
"""
print(sm.accuracy_score(test_y, pred_test_y))   # 0.7333333333333333

print('detection rate: ', sm.precision_score(test_y, pred_test_y, average='macro'))
print('recall rate: ', sm.recall_score(test_y, pred_test_y, average='macro'))
print('f1_score: ', sm.f1_score(test_y, pred_test_y, average='macro'))
print('confusion matrix:\n ', sm.confusion_matrix(test_y, pred_test_y))

print('Classification report:\n', sm.classification_report(test_y, pred_test_y))

"""
detection rate:  0.8095238095238094
recall rate:  0.8095238095238096
f1_score:  0.7333333333333334
confusion matrix: 
  [[5 0 0]
 [0 3 4]
 [0 0 3]]
"""
'''

iris = sd.load_iris()
# print(iris.keys())
"""
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
"""
# print(iris.DESCR)   # 150 samples, 4 features, 3 categories
# print(iris.feature_names)   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# print(iris.target_names)    # ['setosa' 'versicolor' 'virginica']
# print(iris.data.shape)  # (150, 4)
# print(iris.target)
"""
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
"""

# Integrate inputs and outputs into a dataframe
data = pd.DataFrame(iris.data,
                    columns=iris.feature_names)
data['target'] = iris.target
# print(data)

# Organize inputs and outputs
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
# Divide the training set and test set
train_x, test_x, train_y, test_y = ms.train_test_split(x, y,
                                                       test_size=0.1,
                                                       random_state=7,
                                                       stratify=y)

# Build model
model = lm.LogisticRegression(solver='liblinear')
"""
score = ms.cross_val_score(model,   # The model to be validated
                           x, y,     # Full sample data
                           cv=5,    # 5 cross validations
                           scoring='f1_weighted')   # Assessment of indicators
print(score.mean())  # 0.959522933505973
"""

# training model
model.fit(train_x, train_y)
# anticipate
pred_test_y = model.predict(test_x)
# assessment(accuracy)
print('Real category: ', test_y.values)
print('Forecast category: ', pred_test_y)
"""
Real category:  [2 0 0 1 0 2 2 2 1 1 2 1 1 0 0]
Forecast category:  [2 0 0 1 0 2 2 2 1 1 2 1 1 0 0]
"""
print(sm.accuracy_score(test_y, pred_test_y))   # 1.0
print('detection rate: ', sm.precision_score(test_y, pred_test_y, average='macro'))
print('recall rate: ', sm.recall_score(test_y, pred_test_y, average='macro'))
print('f1_score: ', sm.f1_score(test_y, pred_test_y, average='macro'))
print('confusion matrix:\n ', sm.confusion_matrix(test_y, pred_test_y))

print('Classification report:\n', sm.classification_report(test_y, pred_test_y))

"""
detection rate:  1.0
recall rate:  1.0
f1_score:  1.0
confusion matrix:
  [[5 0 0]
 [0 5 0]
 [0 0 5]]
 
Classification report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00         5
           1       1.00      1.00      1.00         5
           2       1.00      1.00      1.00         5

    accuracy                           1.00        15
   macro avg       1.00      1.00      1.00        15
weighted avg       1.00      1.00      1.00        15
"""
