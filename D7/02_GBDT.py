"""
Gradient boosting decision tree
"""

import sklearn.datasets as sd   # data set
import sklearn.model_selection as ms  # Model Selection
import sklearn.ensemble as se   # Integrated Learning
import sklearn.metrics as sm    # Evaluation module


boston = sd.load_boston()

# Organize inputs and outputs
x = boston.data
y = boston.target

# Divide the training set and test set
train_x, test_x, train_y, test_y = ms.train_test_split(x, y, test_size=0.1, random_state=7)

# Build model
model = se.GradientBoostingRegressor(max_depth=6,
                                     n_estimators=400,
                                     random_state=7)
model.fit(train_x, train_y)
pred_train_y = model.predict(train_x)
pred_test_y = model.predict(test_x)
print('training set:', sm.r2_score(train_y, pred_train_y))
print('test set:', sm.r2_score(test_y, pred_test_y))
"""
training set: 0.9999998469170083
test set: 0.9022075571967331
"""
