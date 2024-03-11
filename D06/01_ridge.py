"""
Read SalaryData2.csv
Predicting salary based on work experience (constructing a linear model)
And plot the regression line
"""
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sm

data = pd.read_csv('../data_test/Salary_Data2.csv')
# print(data.head())
"""
   YearsExperience  Salary
0              1.1   39343
1              1.3   46205
2              1.5   37731
3              2.0   43525
4              2.2   39891
"""

# Organize inputs and outputs
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# build a model
model = lm.LinearRegression()
ridge = lm.Ridge(alpha=98)
# train
model.fit(x, y)
ridge.fit(x, y)
# prediction
pred_y = model.predict(x)
pred_ridge = ridge.predict(x)
# tropic
# plt.plot(x, pred_y, color='orangered')
# plt.plot(x, pred_ridge, color='green')
# plt.scatter(x, y)
# plt.show()

# Finding the parameters of alpha in ridge regression
# Find a test set. Assume he hasn't been in training.
"""
test_x = x.iloc[:30:4]
test_y = y[:30:4]

params = np.arange(91, 110, 1)
ret = []
for p in params:
    model = lm.Ridge(alpha=p)
    model.fit(x, y)
    pred_test_y = model.predict(test_x)
    score = sm.r2_score(test_y, pred_test_y)
    print(p, ":--> ", score)
    ret.append(score)

ret = pd.Series(ret, index=params)
print("The best parameter is {}, with a score of {}.".format(ret.idxmax(), ret.max()))
# The best parameter is 98, with a score of 0.9171223161427462.
"""