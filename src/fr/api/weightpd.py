def lr_api(length):
#    import numpy as np
#    from sklearn.model_selection import train_test_split
#    from sklearn.linear_model import LinearRegression
    import sklearn #for predict
    import pickle
    import os

    with open("lr_model.pkl", "rb") as f:
        fish_lr = pickle.load(f)

    lr_w = fish_lr.predict([[length ** 2, length]])
    weight = round(float(lr_w[0]), 3)
    return weight
