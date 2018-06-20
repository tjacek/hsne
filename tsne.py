from sklearn.datasets import fetch_mldata
import time
import knn,markov
from knn import NNGraph
import utils
from sklearn.manifold import TSNE
import numpy as np
import plot

def prepare_hsne(graph_path='mnist/mnist_graph',
                 trans="mnist/scale1/trans.txt",
                 states="mnist/scale1/states.txt"):
    nn_graph=knn.read_nn_graph(graph_path)
    print("nn graph loaded")
    t0=time.time()
    mc=markov.make_eff_markov_chain(nn_graph)
    print("markov chain constructed %d" % (time.time()-t0))
    mc.save(trans,states)

def hsne(dataset_name='MNIST original',landmark_file="mnist/scale1/landmarks.txt",
         influence_file="mnist/scale1/influence.txt",t_file="mnist/scale1/T.txt"):
    landmarks=utils.read_ints(landmark_file)
    print("landmarks loaded")
    sparse_pairs=utils.read_pairs(influence_file)
    print("pairs loaded")
    T=compute_t(landmarks,sparse_pairs)
    utils.save_array(T,t_file,prec='%.2e')
    t_embd=time.time()
    embd=create_embedding(T)
    print("embeding created %d" % (time.time() - t_embd))
    mnist = fetch_mldata(dataset_name)
    plot.plot_embedding(embd,mnist,landmarks,title="beta_threshold=5")

def compute_t(landmarks,sparse_pairs):  
    n_landmarks=len(landmarks)
    print("Number of landmarks %d" % n_landmarks)
    t_sparse=time.time()
    n_states=len(sparse_pairs)
    infl_matrix=utils.to_sparse_matrix(sparse_pairs,n_states,n_landmarks)
    print("sparse matrix created %d" % ( time.time()- t_sparse))
    norm_const=infl_matrix[0].sum()
    infl_matrix/=norm_const
    print(norm_const)
    t_comp=time.time()
    T=markov.get_prob_matrix(infl_matrix)
    print("T matrix computed %d" % (time.time() - t_comp))
    print(T.shape)
    return T

def next_iter(out_file="mnist/scale2",
              landmark_file="mnist/scale1/landmarks.txt",
              t_file="mnist/scale1/T.txt"):
    landmarks=utils.read_ints(landmark_file)
    print("landmarks loaded")
    trans=np.loadtxt(t_file,delimiter=',')
    trans=markov.to_cum_matrix(trans)
    print("trans matrix loaded")
    utils.save_array(trans,out_file+'/trans.txt')
    states_str=",".join([ str(l) for l in landmarks]) 
    utils.save_str(states_str,out_file+'/states.txt')  

def create_embedding(trans):#t_file="T.txt"):
#    trans=np.loadtxt(t_file,delimiter=',')
#    print(trans.shape)
    def norm_helper(row):
        row/=sum(row)
        return row
    trans=np.array([norm_helper(t_i) for t_i in trans])
    P=trans.T +trans
    norm_const=2.0 * float(trans.shape[0])
    P/=norm_const
    embd=TSNE(n_components=2).fit_transform(P)   
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
    markov.compute_influence(mc,landmarks,beta=50)
    print("Time %d" % (time.time() - t0))

#prepare_hsne()
#hsne()
next_iter()