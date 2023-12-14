"""
Linear regression via API provided by sklearn
"""
import pandas as pd
import sklearn.linear_model as lm   # linear model
import matplotlib.pyplot as plt
import sklearn.metrics as sm    # Model Evaluation
import pickle

data = pd.read_csv('../data_test/Salary_Data.csv')
# print(data.head())
"""
   YearsExperience  Salary
0              1.1   39343
1              1.3   46205
2              1.5   37731
3              2.0   43525
4              2.2   39891
"""
# Organize inputs (2D) and outputs (1D)
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Divide the training set and test set


# build a model
model = lm.LinearRegression()
# Training (Analysis)
model.fit(x, y)
# print('w1:', model.coef_)
# print('w0:', model.intercept_)
"""
w1: [9449.96232146]
w0: 25792.200198668696
"""
# predictions
pred_y = model.predict(x)

# tropic
# plt.plot(x, pred_y, color='orangered')
# plt.scatter(x, y)
# plt.show()

# test model
# A portion of the data from the total data is taken as a test set (assuming the test set has not participated in training)
test_x = x.iloc[::4]
test_y = y[::4]
pred_test_y = model.predict(test_x)

# Average absolute error
print(sm.mean_absolute_error(test_y, pred_test_y))  # 4587.366522327393
# mean square error (statistics)
print(sm.mean_squared_error(test_y, pred_test_y))   # 29784216.419621635
# Absolute deviation from median
print(sm.median_absolute_error(test_y, pred_test_y))    # 4895.44536610986
# R2 Score
print(sm.r2_score(test_y, pred_test_y))  # 0.964548495965924

# Saving Models
with open('lr.pickle', 'wb') as f:
    pickle.dump(model, f)

print('Model saved successfully!')