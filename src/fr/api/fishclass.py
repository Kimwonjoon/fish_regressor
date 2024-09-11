import pickle
import os
import sklearn

def get_model_path():
    my_path = __file__
    dir_name = os.path.dirname(my_path)
    model_path = os.path.join(dir_name, "knn_model.pkl")
    return model_path

def knn_api(length, weight):
#    import numpy as np
#    from sklearn.neighbors import KNeighborsClassifier
    with open("knn_model.pkl", "rb") as f:
        fish_knn = pickle.load(f)

    knn_p = fish_knn.predict([[length,weight]])
    fish_type = "빙어"
    if int(knn_p[0]) == 1:
        fish_type = "도미"
    return fish_type
