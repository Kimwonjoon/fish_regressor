import sklearn
import pickle
import os
def get_model_path():
    my_path = __file__
    dir_name = os.path.dirname(my_path)
    model_path = os.path.join(dir_name, "lr_model.pkl")
    return model_path

def lr_api(length):
#    import numpy as np
#    from sklearn.model_selection import train_test_split
#    from sklearn.linear_model import LinearRegression
    path = get_model_path()
    with open(path, "rb") as f:
        fish_lr = pickle.load(f)

    lr_w = fish_lr.predict([[length ** 2, length]])
    weight = round(float(lr_w[0]), 3)
    return weight
