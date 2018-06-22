import time
import utils,knn,markov,tsne,plot
import numpy as np
from scipy.sparse import dok_matrix

def make_dataset(dataset_name="MNIST original",out_path="mnist_d/imgs"):
    dataset=utils.downsample_dataset(dataset_name)
    utils.save_as_img(dataset,out_path)

def make_graph(dataset_name="mnist_d/imgs",out_path="mnist_d/nn_graph",k=100):
    dataset=utils.read_as_dataset(dataset_name)
    print("dataset loaded")
    knn.save_nn_graph(dataset,out_path)

def prepare_hsne(graph_path='mnist_d/nn_graph',
                 trans="mnist_d/scale1/trans.txt",
                 states="mnist_d/scale1/states.txt"):
    nn_graph=knn.read_nn_graph(graph_path)
    print("nn graph loaded")
    t0=time.time()
    mc=markov.make_eff_markov_chain(nn_graph)
    print("markov chain constructed %d" % (time.time()-t0))
    mc.save(trans,states)

def hsne(graph_path='mnist_d/nn_graph',landmark_file="mnist_d/scale1/landmarks.txt",
         influence_file="mnist_d/scale1/influence.txt",t_file="mnist_d/scale1/T.txt",
         weights_in=None,weights_out="mnist_d/scale2/W.txt"):
    landmarks=utils.read_ints(landmark_file)
    print("landmarks loaded")
    sparse_pairs=utils.read_pairs(influence_file)
    print("pairs loaded %d" % len(sparse_pairs))
    if(weights_in is None):
        n_points=len(sparse_pairs)
        W=dok_matrix(np.ones((n_points,1)),dtype=np.float32)
    else:
        W=np.loadtxt(weights_file)
    T,W_next=tsne.compute_t(landmarks,sparse_pairs,W)
    t_embd=time.time()
    embd=tsne.create_embedding(T)
    print("embeding created %d" % (time.time() - t_embd))
    #mnist = fetch_mldata(dataset_name)
    nn_graph=knn.read_nn_graph(graph_path)
    plot.plot_embedding(embd,nn_graph.target,landmarks,title="beta_threshold=1.5")
    utils.save_array(T,t_file,prec='%.2e')
    utils.save_array(W_next,weights_out)

def next_iter(out_file="mnist_d/scale2",
              landmark_file="mnist_d/scale1/landmarks.txt",
              t_file="mnist_d/scale1/T.txt"):
    landmarks=utils.read_ints(landmark_file)
    print("landmarks loaded")
    trans=np.loadtxt(t_file,delimiter=',')
    trans=markov.to_cum_matrix(trans)
    print("trans matrix loaded")
    utils.save_array(trans,out_file+'/trans.txt')
    states_str=",".join([ str(l) for l in landmarks]) 
    utils.save_str(states_str,out_file+'/states.txt') 

#prepare_hsne()
hsne()
#next_iter()