import numpy as np 
import tsne,utils,plot
from sklearn.datasets import fetch_mldata

def reconstruct(matrix_path,embd_path,out_path):
    embd =utils.read_object(embd_path)
    infl=utils.read_object(matrix_path)
    n_points=infl.shape[0]
    n_embd=embd.shape[0]
    def recon_helper(i):
        print(i)
        embd_weighted=np.array([ infl[i,j]*embd[j] for j in range(n_embd)])
        rec_i=np.sum(embd_weighted,axis=0)
        print(rec_i.shape)
        return rec_i
    reconstruction=np.array([recon_helper(i) for i in range(n_points)])
    utils.save_object(reconstruction,out_path)

def make_embd(scale_path="mnist/scale1"):
    in_file=scale_path+"/T.txt"
    out_file=scale_path+"/emd"
    trans=utils.read_object(in_file)
    T=tsne.create_embedding(trans)
    utils.save_object(T,out_file)

def show_embd(scale_path="mnist/scale1",dataset_name="MNIST original",threshold=1.5,embd_path=None):
    if(embd_path is None):
        in_file=scale_path+"/emd"	
    else:
        in_file=embd_path	
    X=utils.read_object(in_file)
    landmark_file= scale_path+"/landmarks.txt"
    landmarks=utils.read_ints(landmark_file)
    mnist = fetch_mldata(dataset_name)
    title="beta_threshold="+str(threshold)
    plot.plot_embedding(X,mnist.target,landmarks,title=title)

def rec_matrix(scale_paths,out_path):
    def inf_matrix(scale_i):
        landmarks=utils.read_ints(scale_i +"/landmarks.txt") 
        sparse_pairs=utils.read_pairs(scale_i +"/influence.txt") 
        return tsne.make_influence_matrix(landmarks,sparse_pairs)

    infl_matrixs=[inf_matrix(scale_i)
                   for scale_i in scale_paths]
    rec_matrix=infl_matrixs[0]
    for infl_i in infl_matrixs[1:]:
        rec_matrix=rec_matrix*infl_i	
    utils.save_object(rec_matrix.todense(),out_path)
    #print(rec_matrix.shape)
    #for infl_i in infl_matrixs:
    #    print(infl_i.shape)

#make_embd(scale_path="/mnist/scale1")
#show_embd()
scales=["mnist/scale1","mnist/scale2","mnist/scale3"]
rec_matrix(scales,"mnist/rec_")
#reconstruct("mnist/rec","mnist/scale2/emd",out_path="mnist/embd")
#show_embd(scale_path="mnist/scale1",embd_path="mnist/embd")