from sklearn.datasets import fetch_mldata
import time
import knn,markov
from knn import NNGraph
import utils

def make_markov(graph_path):
    nn_graph=knn.read_nn_graph(graph_path)
    mc=markov.make_eff_markov_chain(nn_graph)#make_markov_chain(nn_graph)
    print(mc.states)
    mc.save('trans.txt','states.txt')

def select_landmarks(dataset,in_file='landmarks.txt',out_file='landmarks'):
    landmarks=utils.read_ints(in_file)	
    utils.save_as_img(dataset.data,dataset.target,out_path=out_file,new_shape=(28,28),selected=landmarks)

#iteration('mnist_graph')
#select_landmarks('landmarks.txt')
mnist = fetch_mldata('MNIST original')
select_landmarks(mnist)