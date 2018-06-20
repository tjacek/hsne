import matplotlib.pyplot as plt
from matplotlib import offsetbox
import numpy as np

def plot_embedding(X,digits,landmarks,fraction=3,title=None):
    print(landmarks)
    y = [digits.target[l] 
            for l in landmarks]
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        if(i%fraction == 0):
            plt.text(X[i, 0], X[i, 1], str(y[i]),
                   color=plt.cm.Set1(y[i] / 10.),
                   fontdict={'weight': 'bold', 'size': 9})

    #if hasattr(offsetbox, 'AnnotationBbox'):
    #    # only print thumbnails with matplotlib > 1.0
    #    shown_images = np.array([[1., 1.]])  # just something big
    #    for i in range(digits.data.shape[0]):
    #        dist = np.sum((X[i] - shown_images) ** 2, 1)
    #        if np.min(dist) < 4e-3:
    #            # don't show points that are too close
    #            continue
    #        shown_images = np.r_[shown_images, [X[i]]]
    #        imagebox = offsetbox.AnnotationBbox(
    #            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
    #            X[i])
    #        ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()