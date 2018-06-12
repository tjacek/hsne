from sklearn.datasets import fetch_mldata
from sklearn.neighbors import LSHForest,NearestNeighbors
import utils
import time

class NNGraph(object):
    def __init__(self,names,distances):
        self.names=names
        self.distances=distances

def make_nn_graph(X,k=1000):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
#    nbrs=LSHForest(n_estimators=20, n_candidates=200,n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)
    print(indices.shape)
    return NNGraph(indices,distances)

def save_nn_graph(data,out_path):
    t0=time.time()
    print(t0)
    nn_graph=make_nn_graph(data)
    print(time.time()-t0)
    utils.save_object(nn_graph,out_path)

mnist = fetch_mldata('MNIST original')
print( mnist.data.shape)
save_nn_graph(mnist.data,"mnist_graph")