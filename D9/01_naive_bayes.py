"""
Plain Bayes (math.)
"""
import pandas as pd
import sklearn.model_selection as ms    # Model Selection
import sklearn.naive_bayes as nb    # Plain Bayes (math.)
import sklearn.metrics as sm    # valuation

data = pd.read_csv('../data_test/multiple1.txt',
                   header=None,
                   names=['x1', 'x2', 'y'])
# print(data.head())
"""
     x1    x2  y
0  8.73  0.31  2
1  4.71 -0.42  3
2  4.58  6.18  1
3  9.38  2.18  2
4  4.78  5.28  1
"""

# Organize inputs and outputs
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Divide the training set and test set
train_x, test_x, train_y, test_y = ms.train_test_split(x, y,
                                                       test_size=0.1,
                                                       random_state=7,
                                                       stratify=y)

# Build a model
model = nb.GaussianNB()
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.classification_report(test_y, pred_test_y))
"""
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00        10
           2       1.00      1.00      1.00        10
           3       1.00      1.00      1.00        10

    accuracy                           1.00        40
   macro avg       1.00      1.00      1.00        40
weighted avg       1.00      1.00      1.00        40
"""