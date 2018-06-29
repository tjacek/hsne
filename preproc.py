import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn.datasets.base import Bunch
import knn

def pca_preproc(dataset_name="MNIST original"):
    dataset=fetch_mldata(dataset_name)
    n_dim=dataset.data.shape[1]
    transform=PCA(n_components=n_dim)
    t0=time.time()
    transformed=transform.fit_transform(dataset.data)
    print("PCA transform %d" % (time.time()-t0))
    n_feats=find_suff_size(transform.explained_variance_ratio_ )
    reduced=transformed[:,:n_feats]
    return Bunch(data=reduced,target=dataset.target)

def find_suff_size(expl_variance,threshold=0.95):
    var=0.0
    for i,var_i in enumerate(expl_variance):
        var+=var_i
        if(var>=threshold):
        	return i
    return len(list(expl_variance))

if __name__ == "__main__": 
    dataset=pca_preproc(dataset_name="MNIST original")
    knn.save_nn_graph(dataset,"mnist_pca/graph")