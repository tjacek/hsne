from sklearn.datasets import fetch_mldata
import time
import knn,markov
from knn import NNGraph
import utils
from sklearn.manifold import TSNE
import numpy as np

def make_markov(graph_path):
    nn_graph=knn.read_nn_graph(graph_path)
    mc=markov.make_eff_markov_chain(nn_graph)#make_markov_chain(nn_graph)
    print(mc.states)
    mc.save('trans.txt','states.txt')

def select_landmarks(dataset,in_file='landmarks.txt',out_file='landmarks'):
    landmarks=utils.read_ints(in_file)	
    utils.save_as_img(dataset.data,dataset.target,out_path=out_file,new_shape=(28,28),selected=landmarks)

def compute_tsne(t_file="T.txt"):
    trans=np.loadtxt(t_file,delimiter=',')
    print(trans.shape)
    def norm_helper(row):
        row/=sum(row)
        return row
    trans=np.array([norm_helper(t_i) for t_i in trans])
    P=trans.T +trans
    norm_const=2.0 * float(trans.shape[0])
    P/=norm_const
    embd=TSNE(n_components=2).fit_transform(P)   
    print(embd.shape)

def compute_t(landmark_file,influence_file):
    landmarks=utils.read_ints(landmark_file)
    sparse_pairs=utils.read_pairs(influence_file)
    print("pairs loaded")
    n_landmarks=len(landmarks)
    n_states=len(sparse_pairs)
    infl_matrix=utils.to_sparse_matrix(sparse_pairs,n_states,n_landmarks)
    norm_const=infl_matrix[0].sum()
    infl_matrix/=norm_const
    print(norm_const)
    markov.get_prob_matrix(infl_matrix)

def compute_influence(graph_path,landmark_file):
    nn_graph=knn.read_nn_graph(graph_path)
    print("nn graph loaded")
    mc=markov.make_eff_markov_chain(nn_graph)
    print("markov chain built")
    landmarks=utils.read_ints(landmark_file)
    t0=time.time()
    markov.compute_influence(mc,landmarks,beta=50)
    print("Time %d" % (time.time() - t0))

compute_tsne("T.txt")#'landmarks.txt','influence.txt')
#iteration('mnist_graph')
#select_landmarks('landmarks.txt')
#mnist = fetch_mldata('MNIST original')
#select_landmarks(mnist)