import os,time
import utils,knn,markov,tsne,plot
import numpy as np
from scipy import sparse
from knn import NNGraph
from sklearn.datasets import fetch_mldata

#def make_dataset(dataset_name="MNIST original",out_path="mnist_d/imgs"):
#    dataset=utils.downsample_dataset(dataset_name)
#    utils.save_as_img(dataset,out_path)

#def make_graph(dataset_name="mnist_d/imgs",out_path="mnist_d/nn_graph",k=100):
#    dataset=utils.read_as_dataset(dataset_name)
#    print("dataset loaded")
#    knn.save_nn_graph(dataset,out_path)

def prepare_hsne(graph_path='mnist_d/nn_graph',
                 scale_path='mnist_d/scale1'):
    os.mkdir(scale_path)
    trans=scale_path+ "/trans.txt"
    states=scale_path+ "/states.txt"
    nn_graph=knn.read_nn_graph(graph_path)
    print("nn graph loaded")
    t0=time.time()
    mc=markov.make_eff_markov_chain(nn_graph)
    print("markov chain constructed %d" % (time.time()-t0))
    mc.save(trans,states)

def hsne(dataset_name="MNIST original",
         scale_path='mnist_d/scale1',
         weights_in=None):

    landmarks,sparse_pairs=load_hsne(scale_path)
   
    W=get_weights(weights_in,sparse_pairs)
    T,W_next=tsne.compute_t(landmarks,sparse_pairs,W)
    t_embd=time.time()
    embd=tsne.create_embedding(T)
    print("embeding created %d" % (time.time() - t_embd))
    mnist = fetch_mldata(dataset_name)
    #nn_graph=knn.read_nn_graph(graph_path)
    plot.plot_embedding(embd,mnist.target,landmarks,title="beta_threshold=5")
    save_hsne(T,W_next,scale_path)

def load_hsne(scale_path):
    landmark_file= scale_path+"/landmarks.txt"
    print(landmark_file)
    landmarks=utils.read_ints(landmark_file)
    print("landmarks loaded")
    influence_file=scale_path+"/influence.txt"
    sparse_pairs=utils.read_pairs(influence_file)
    print("pairs loaded %d" % len(sparse_pairs))
    return landmarks,sparse_pairs

def save_hsne(T,W_next,scale_path):
    t_file=scale_path+"/T.txt"
    weights_out=scale_path+"/W.txt"
    utils.save_object(T,t_file)
    utils.save_array(W_next,weights_out)

def next_iter(in_scale="mnist_d/scale1",out_scale="mnist_d/scale2" ):
#    os.mkdir(out_scale)
    landmarks,trans=load_iter(in_scale)

    trans=markov.to_cum_matrix(trans)   
    states_str=",".join([ str(l) for l in landmarks])
    save_iter(trans,states_str,out_scale)

def load_iter(in_scale):
    landmark_file=in_scale+"/landmarks.txt"
    landmarks=utils.read_ints(landmark_file)
    print("landmarks loaded")
    t_file=in_scale+"/T.txt"
    trans=utils.read_object(t_file)#np.loadtxt(t_file,delimiter=',')
    print("trans matrix loaded")
    return landmarks,trans

def save_iter(trans,states_str,out_scale):
    trans_file=out_scale+"/trans.txt"
    utils.save_array(trans,trans_file)
    states_file=out_scale+"/states.txt"
    utils.save_str(states_str,states_file) 

def get_weights(weights_in,sparse_pairs):
    if(weights_in is None):
        n_points=len(sparse_pairs)
        W=sparse.dok_matrix(np.ones((n_points,1)),dtype=np.float32)
    else:
        weights_file=weights_in+"/W.txt"
        W=np.loadtxt(weights_file,delimiter=',')
        W=np.expand_dims(W,axis=1)
        W=sparse.dok_matrix(W)
    return W

#prepare_hsne(graph_path='mnist/graph',scale_path='mnist/scale1')
#hsne(scale_path="mnist/scale1",weights_in=None)#"mnist/scale1")
next_iter(in_scale="mnist/scale1",out_scale="mnist/scale2")