import matplotlib.pyplot as plt
from matplotlib import offsetbox
import numpy as np

def plot_embedding(X,cats,landmarks,title=None):
#    print(landmarks)
    n_points=X.shape[0]
    fraction=get_fraction(n_points)
    print("Fraction %d" % fraction)
    y = [cats[l] 
            for l in landmarks]
    print("Unique categories")
    print(np.unique(y))
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(n_points):
        if( (i%fraction) == 0):
            plt.text(X[i, 0], X[i, 1], str(y[i]),
                   color=plt.cm.Set3( float(y[i]) / 10.),
                   fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()

def get_fraction(n_points,max_points=3000):
    if(n_points>max_points):
        return int(n_points/max_points)
    else:
        return 1
