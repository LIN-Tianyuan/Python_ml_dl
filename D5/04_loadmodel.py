"""
Load the model and perform the prediction
"""
import pickle

with open('./lr.pickle', 'rb') as f:
    model = pickle.load(f)

print(model.predict([[1.1], [2.2]]))    # [36187.15875227 46582.11730587]
