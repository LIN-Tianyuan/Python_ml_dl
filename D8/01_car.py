"""
Predicting the class of a small car based on a set of characteristics
A good set of data can build a good model, a bad set of data must not build a good model,
when in the business of categorization,
the first thing to look at is whether the number of samples in each category is balanced or not

Sample category equalization: upsampling, downsampling, Insufficient samples, weights.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing as sp  # Data preprocessing
import sklearn.ensemble as se   # Integrated Learning
import sklearn.model_selection as ms    # Model Selection

data = pd.read_csv('../data_test/car.txt',
                   header=None)
# print(data.head())
"""
       0      1  2  3      4     5      6
0  vhigh  vhigh  2  2  small   low  unacc
1  vhigh  vhigh  2  2  small   med  unacc
2  vhigh  vhigh  2  2  small  high  unacc
3  vhigh  vhigh  2  2    med   low  unacc
4  vhigh  vhigh  2  2    med   med  unacc
"""
# print(data.dtypes)
"""
0    object
1    object
2    object
3    object
4    object
5    object
6    object
dtype: object
"""

train_data = pd.DataFrame()
encoders = {}
for i in data:
    encoder = sp.LabelEncoder()
    res = encoder.fit_transform(data[i])
    train_data[i] = res
    encoders[i] = encoder

# print(train_data)
"""
      0  1  2  3  4  5  6
0     3  3  0  0  2  1  2
1     3  3  0  0  2  2  2
2     3  3  0  0  2  0  2
3     3  3  0  0  1  1  2
4     3  3  0  0  1  2  2
...  .. .. .. .. .. .. ..
1723  1  1  3  2  1  2  1
1724  1  1  3  2  1  0  3
1725  1  1  3  2  0  1  2
1726  1  1  3  2  0  2  1
1727  1  1  3  2  0  0  3
"""

# Organize inputs and outputs
train_x = train_data.iloc[:, :-1]
train_y = train_data.iloc[:, -1]
# Build a model(Grid Search)
params = {'max_depth': np.arange(5, 18),
          'n_estimators': np.arange(100, 801, 50),
          'criterion': ['gini', 'entropy']}
sub_model = se.RandomForestClassifier()
model = ms.GridSearchCV(sub_model, params, cv=3)
"""
# Build a model
model = se.RandomForestClassifier(max_depth=6,
                                  n_estimators=400,
                                  random_state=7,
                                  class_weight='balanced')
"""
'''
# validation curve
params = np.arange(100, 1001, 100)
train_score, test_score = ms.validation_curve(model,
                                              train_x, train_y,
                                              param_name='n_estimators',
                                              param_range=params,
                                              cv=5)
avg_score = test_score.mean(axis=1)
plt.plot(params, avg_score, 'o-')
plt.show()


# learning curve
params = np.arange(0.1, 1.1, 0.1)
train_size, train_score, test_score = ms.learning_curve(model,
                                                        train_x, train_y,
                                                        train_sizes=params,
                                                        cv=5)
avg_score = test_score.mean(axis=1)
plt.plot(params, avg_score, 'o-')
plt.show()
'''

# print(len(data))    # 1728
# print(data[6].value_counts())
"""
unacc    1210
acc       384
good       69
vgood      65
Name: 6, dtype: int64
"""


# Train
model.fit(train_x, train_y)
print('Optimal parameter combinations: ', model.best_params_)
print('Best score: ', model.best_score_)

best_model = model.best_estimator_

# Predict
test_data = [['high', 'med', '5more', '4', 'big', 'low', 'unacc'],
             ['high', 'high', '4', '4', 'med', 'med', 'acc'],
             ['low', 'low', '2', '4', 'small', 'high', 'good'],
             ['low', 'med', '3', '4', 'med', 'high', 'vgood']]
test_data = pd.DataFrame(test_data)
for i in test_data:
    encoder = encoders[i]
    res = encoder.transform(test_data[i])
    test_data[i] = res
# print(test_data)
"""
   0  1  2  3  4  5  6
0  0  2  3  1  0  1  2
1  0  0  2  1  1  2  0
2  1  1  0  1  2  0  1
3  1  2  1  1  1  0  3
"""
test_x = test_data.iloc[:, :-1]
test_y = test_data.iloc[:, -1]

# pred_test_y = model.predict(test_x)
pred_test_y = best_model.predict(test_x)
# Evaluation
# print('Real category: ', test_y.values)
# print('Forecast category: ', pred_test_y)
"""
Real category:  [2 0 1 3]
Forecast category:  [2 0 0 0]
"""
print('Real category: ', encoders[6].inverse_transform(test_y.values))
print('Forecast category: ', encoders[6].inverse_transform(pred_test_y))
"""
Real category:  ['unacc' 'acc' 'good' 'vgood']
Forecast category:  ['unacc' 'acc' 'acc' 'acc']
"""
print(pred_test_y)
print(best_model.predict_proba(test_x))
"""
Optimal parameter combinations:  {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 300}
Best score:  0.8003472222222222
Real category:  ['unacc' 'acc' 'good' 'vgood']
Forecast category:  ['unacc' 'acc' 'acc' 'vgood']
[2 0 0 3]
[[0.16168013 0.02472233 0.80120498 0.01239255]
 [0.50063121 0.06291675 0.43151879 0.00493325]
 [0.39702497 0.28555756 0.232047   0.08537047]
 [0.38065097 0.14449845 0.09099375 0.38385683]]
"""