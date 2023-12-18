"""
polynomial regression:
1. Extended characterization (addition of higher order terms).
2. Give the expanded features to the linear regression analyzer.
"""
import pandas as pd
import sklearn.pipeline as pl   # Pipeline Module
import sklearn.preprocessing as sp  # Data preprocessing
import sklearn.linear_model as lm   # linear model
import matplotlib.pyplot as plt

data = pd.read_csv('../data_test/Salary_Data.csv')

# Organize inputs and outputs
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# build a model
model = pl.make_pipeline(sp.PolynomialFeatures(3),
                         lm.LinearRegression())
# train
model.fit(x, y)
# predict
pred_y = model.predict(x)
# tropic
plt.plot(x, pred_y, color='orangered')
plt.scatter(x, y)
plt.show()
"""
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

data = pd.read_csv('../data_test/Salary_Data.csv')

# Organize inputs and outputs
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

x['x2'] = x['YearsExperience'] ** 2
x['x3'] = x['YearsExperience'] ** 3

model = lm.LinearRegression()
model.fit(x, y)
# print(model.coef_)  # [-718.70841416 2099.35194631 -122.91541434]
# print(model.intercept_)  # 38863.07185016371

pred_y = model.predict(x)
plt.plot(x['YearsExperience'], pred_y, color="orangered")
plt.scatter(x['YearsExperience'], y)
plt.show()
"""