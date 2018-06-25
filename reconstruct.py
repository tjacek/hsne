import numpy as np 
import tsne,utils

def make_embd(scale_path="/mnist/scale1"):
    in_file=scale_path+"/T.txt"
    out_file=scale_path+"/emd"
    trans=utils.read_object(in_file)
    T=tsne.create_embedding(trans)
    utils.save_object(T,out_file)

make_embd(scale_path="/mnist/scale1")