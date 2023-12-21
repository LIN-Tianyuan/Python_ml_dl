"""
Boston Home Price Forecast
"""

import sklearn.datasets as sd  # data set
import sklearn.model_selection as ms  # Model Selection
import sklearn.linear_model as lm   # linear model
import sklearn.metrics as sm    # Evaluation module
import sklearn.pipeline as pl   # data pipeline
import sklearn.preprocessing as sp  # Data preprocessing
import sklearn.tree as st  # decision tree
import pandas as pd
import matplotlib.pyplot as plt

boston = sd.load_boston()
# print(boston.keys())    # dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename', 'data_module'])
# print(boston.filename)  # boston_house_prices.csv
# print(boston.DESCR)   # 506 samples, 13 columns of characteristics, 1 median price
# print(boston.feature_names) # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
# print(boston.data.shape)    # (506, 13)
# print(boston.target.shape)  # (506,)

# Organize inputs and outputs
x = boston.data
y = boston.target

# Divide the training set and test set(The data in the training set must not be sequential)
train_x, test_x, train_y, test_y = ms.train_test_split(x, y,    # Data to be classified
                                                       test_size=0.2, # Test set share
                                                       random_state=7)  # Random seed


# Build model
# Linear regression, ridge regression, polynomial regression
# Evaluating R2 for training and test sets
def get_model(model, name):
    print('----------', name, '----------')
    model.fit(train_x, train_y)
    pred_train_y = model.predict(train_x)
    pred_test_y = model.predict(test_x)
    print('training set:', sm.r2_score(train_y, pred_train_y))
    print('test set:', sm.r2_score(test_y, pred_test_y))


model_dict = {'Linear regression': lm.LinearRegression(),
              'Ridge regression': lm.Ridge(),
              'Polynomial regression': pl.make_pipeline(sp.PolynomialFeatures(2),
                                                        lm.LinearRegression())}
# for name, model in model_dict.items():
#     get_model(model, name)

"""
---------- Linear regression ----------
training set: 0.7698532963729757
test set: 0.578541547276341

---------- Ridge regression ----------
training set: 0.7681931875788315
test set: 0.5703641157344462

---------- Polynomial regression ----------
training set: 0.9336239312238316
test set: 0.6170018549388412
"""

# single-tree regression
model = st.DecisionTreeRegressor(max_depth=4, random_state=7)
model.fit(train_x, train_y)
pred_train_y = model.predict(train_x)
pred_test_y = model.predict(test_x)
print('Single-tree training sets:', sm.r2_score(train_y, pred_train_y))
print('Single tree test set:', sm.r2_score(test_y, pred_test_y))
"""
Single-tree training sets: 0.9466499293522961
Single tree test set: 0.7006779545917512
"""

# Importance of features
fi = model.feature_importances_
print(fi)
fi = pd.Series(fi,
               index=boston.feature_names)
fi = fi.sort_values(ascending=False)
print(fi)

# plt.bar(fi.index, fi.values)


# Decision Tree Visualization
st.plot_tree(model, fontsize=10,
             feature_names=boston.feature_names,
             filled=True)
plt.show()

