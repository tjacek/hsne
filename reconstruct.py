import numpy as np 
import tsne,utils,plot
from sklearn.datasets import fetch_mldata

def make_embd(scale_path="mnist/scale1"):
    in_file=scale_path+"/T.txt"
    out_file=scale_path+"/emd"
    trans=utils.read_object(in_file)
    T=tsne.create_embedding(trans)
    utils.save_object(T,out_file)

def show_embd(scale_path="mnist/scale1",dataset_name="MNIST original",threshold=1.5):
    in_file=scale_path+"/emd"	
    X=utils.read_object(in_file)
    landmark_file= scale_path+"/landmarks.txt"
    landmarks=utils.read_ints(landmark_file)
    mnist = fetch_mldata(dataset_name)
    title="beta_threshold="+str(threshold)
    plot.plot_embedding(X,mnist.target,landmarks,title="beta_threshold=5")

#make_embd(scale_path="/mnist/scale1")
show_embd()