from sklearn.datasets import fetch_mldata
import time
import knn,markov
from knn import NNGraph
import utils
from sklearn.manifold import TSNE
import numpy as np
import plot

def compute_t(landmarks,sparse_pairs,W):  
    infl_matrix=make_influence_matrix(landmarks,sparse_pairs)
    t_comp=time.time()
    T=markov.get_prob_matrix(infl_matrix,W)
    print("T matrix computed %d" % (time.time() - t_comp))
    print(T.shape)
    def norm_helper(row):
        row/=sum(row)
        return row
    T=np.array([norm_helper(t_i) for t_i in T])
    print("norm %d" % check_norm(T))
    W_next=(W.transpose()*infl_matrix).todense()
    print("W_next"+str(type(W_next)))
    print(W_next.shape)
 #   W_next=np.expand_dims(W_next,axis=0)
 #   print(W_next.shape)
    return T,W_next

def make_influence_matrix(landmarks,sparse_pairs):
    n_landmarks=len(landmarks)
    print("Number of landmarks %d" % n_landmarks)
    t_sparse=time.time()
    n_states=len(sparse_pairs)
    infl_matrix=utils.to_sparse_matrix(sparse_pairs,n_states,n_landmarks)
    print("sparse matrix created %d" % ( time.time()- t_sparse))
    norm_const=infl_matrix[0].sum()
    infl_matrix/=norm_const
    print("Norm const %d" % norm_const)
    return infl_matrix

def check_norm(T):
    s=np.sum(T,axis=1)
    for s_i in s:
        if(  (1.0-s_i)>0.01 ):
            return False
    return sum(s) 

def create_embedding(trans):

    P=trans.T +trans
    norm_const=2.0 * float(trans.shape[0])
    P/=norm_const
    embd=TSNE(n_components=2,perplexity=20).fit_transform(P)   
    return embd

def select_landmarks(dataset,in_file='landmarks.txt',out_file='landmarks'):
    landmarks=utils.read_ints(in_file)	
    utils.save_as_img(dataset.data,dataset.target,out_path=out_file,new_shape=(28,28),selected=landmarks)

def compute_influence(graph_path,landmark_file):
    nn_graph=knn.read_nn_graph(graph_path)
    print("nn graph loaded")
    mc=markov.make_eff_markov_chain(nn_graph)
    print("markov chain built")
    landmarks=utils.read_ints(landmark_file)
    t0=time.time()
    markov.compute_influence(mc,landmarks,beta=100)
    print("Time %d" % (time.time() - t0))