def knn_api(length, weight):
#    import numpy as np
#    from sklearn.neighbors import KNeighborsClassifier
    import pickle
    import os
    import sklearn

    with open("knn_model.pkl", "rb") as f:
        fish_knn = pickle.load(f)

    knn_p = fish_knn.predict([[length,weight]])
    fish_type = "빙어"
    if int(knn_p[0]) == 1:
        fish_type = "도미"
    return fish_type
