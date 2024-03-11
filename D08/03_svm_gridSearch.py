"""
Implementing SVM using grid search
"""
import pandas as pd
import sklearn.model_selection as ms    # Model Selection
import sklearn.svm as svm   # support vector machine
import sklearn.metrics as sm    # valuation

data = pd.read_csv('../data_test/multiple2.txt',
                   header=None,
                   names=['x1', 'x2', 'y'])

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

train_x, test_x, train_y, test_y = ms.train_test_split(x, y,
                                                       test_size=0.1,
                                                       random_state=7,
                                                       stratify=y)
params = [{'kernel': ['linear'], 'C':[1, 10, 100]},
          {'kernel': ['poly'], 'degree': [2, 3], 'C': [1, 10, 100]},
          {'kernel': ['rbf'], 'gamma': [0.1, 0.01, 0.001], 'C': [1, 10, 100]}]
sub_model = svm.SVC()
model = ms.GridSearchCV(sub_model,
                        params,
                        cv=3)
model.fit(x, y)
print('Optimal model parameters: ', model.best_params_)
print('Best score: ', model.best_score_)

best_model = model.best_estimator_

pred_test_y = best_model.predict(test_x)
print(sm.classification_report(test_y,
                               pred_test_y))

"""
Optimal model parameters:  {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
Best score:  0.96
              precision    recall  f1-score   support

           0       0.94      1.00      0.97        15
           1       1.00      0.93      0.97        15

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
"""