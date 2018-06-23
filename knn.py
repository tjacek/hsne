from sklearn.datasets import fetch_mldata
from sklearn.neighbors import LSHForest,NearestNeighbors
import time
import markov,utils

class NNGraph(object):
    def __init__(self,names,distances,target):
        self.names=names
        self.distances=distances
        self.target=target

    def __len__(self):
        return len(self.names)
        
    def __getitem__(self,i):
        return self.names[i],self.distances[i]

def make_nn_graph(dataset,k=100):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(dataset.data)
#    nbrs=LSHForest(n_estimators=20, n_candidates=200,n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(dataset.data)
    print(indices.shape)
    return NNGraph(indices,distances,dataset.target)

def read_nn_graph(in_path):
    t0=time.time()
    nn_graph=utils.read_object(in_path)
    print(time.time()-t0)
    return nn_graph

def save_nn_graph(data,out_path):
    t0=time.time()
    nn_graph=make_nn_graph(data)
    print(time.time()-t0)
    utils.save_object(nn_graph,out_path)

if __name__ == "__main__": 
    dataset=fetch_mldata("MNIST original")
    save_nn_graph(dataset,"mnist/graph")