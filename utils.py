import numpy as np
import cv2
import pickle,os,re
#from sets import Set 
from scipy.sparse import dok_matrix
import sklearn.datasets.base
from sklearn.datasets import fetch_mldata

def downsample_dataset(dataset_name,factor=10):
    dataset=fetch_mldata(dataset_name)
    examples=[ example_i
               for i,example_i in enumerate(dataset.data)
                 if((i % factor) ==0)]
    target = [ example_i
               for i,example_i in enumerate(dataset.target)
                 if((i % factor) ==0)]
    return sklearn.datasets.base.Bunch(data=examples, target=target)

def save_as_img(dataset,out_path,new_shape=(28,28),selected=None):
    if(not selected is None):
        selected=Set(selected)
    def save_helper(i,img_i):        
        img_i=np.reshape(img_i,new_shape)
        cat_i=str(int(dataset.target[i]))
        name_i=out_path+'/'+str(i)+'_'+ cat_i +'.png'
        cv2.imwrite(name_i,img_i)
        print(name_i)
    for i,img_i in enumerate(dataset.data):
        if((selected is None) or (i in selected)):
            save_helper(i,img_i)

def read_as_dataset(in_path):
    def read_helper(filename_i):
        img_path_i=in_path+'/'+filename_i 
        img_i=cv2.imread(img_path_i,0)
        print(img_i.shape)
        img_i=img_i.flatten()
        cat_i=int(extract_int(filename_i)[1])
        return img_i,cat_i
    imgs=[ read_helper(filename_i)
            for filename_i in os.listdir(in_path)]
    data=np.array([ img_i[0] for img_i in imgs])
    target=np.array([ img_i[1] for img_i in imgs])
    return sklearn.datasets.base.Bunch(data=data, target=target)

def extract_int(str_i):
    return re.findall('\d+', str_i )    
        
def read_ints(filename):
    with open(filename) as f:
        raw_ints = f.readlines()
    return [ int(raw_i) for raw_i in raw_ints]    

def save_str(txt,out_path):
    text_file = open(out_path, "w")
    text_file.write(txt)
    text_file.close()

def save_array(arr,out_path,prec='%.4e'):
    np.savetxt(out_path, arr, fmt=prec, delimiter=',', newline='\n')

def save_object(nn,path):
    file_object = open(path,'wb')
    pickle.dump(nn,file_object)
    file_object.close()

def read_object(path):
    file_object = open(path,'rb')
    obj=pickle.load(file_object)  
    file_object.close()
    return obj

def to_sparse_matrix(sparse_pairs,n_states,n_landmarks):
    infl_matrix=dok_matrix((n_states, n_landmarks), dtype=np.float32)
    for i,pairs_i in enumerate(sparse_pairs):
        for j,value_j in pairs_i:
            infl_matrix[i,j]=value_j
    return infl_matrix 

def read_pairs(filename):
    with open(filename) as f:
        lines = f.readlines()
    def parse_pair(pair):
        key,value=pair.split(",")
        return int(key),float(value)
    def parse_line(line):
        pairs=line.split(")(")
        pairs[0]=pairs[0].replace("(","")
        pairs[-1]=pairs[-1].replace(")","")
        return [ parse_pair(pair_i) for pair_i in pairs]
    sparse_pairs=[parse_line(line_i)  for line_i in lines]
    return sparse_pairs