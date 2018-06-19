import numpy as np
import numpy
import cv2
import pickle
from sets import Set 

def save_as_img(dataset,labels,out_path,new_shape=(28,28),selected=None):
    if(not selected is None):
        selected=Set(selected)
    def save_helper(i,img_i):        
        img_i=np.reshape(img_i,new_shape)
        cat_i=str(int(labels[i]))
        name_i=out_path+'/'+str(i)+'_'+ cat_i +'.png'
        cv2.imwrite(name_i,img_i)
        print(name_i)
    for i,img_i in enumerate(dataset):
        if((selected is None) or (i in selected)):
            save_helper(i,img_i)

def read_sparse_matrix(filename):
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

def read_ints(filename):
    with open(filename) as f:
        raw_ints = f.readlines()
    return [ int(raw_i) for raw_i in raw_ints]    

def save_array(arr,out_path,prec='%.4e'):
    np.savetxt(out_path, arr, fmt=prec, delimiter=',', newline='\n')

def save_object(nn,path):
    file_object = open(path,'wb')
    pickle.dump(nn,file_object)
    file_object.close()

def read_object(path):
    file_object = open(path,'r')
    obj=pickle.load(file_object)  
    file_object.close()
    return obj