import numpy
import cv2
import pickle
import knn

def save_as_img(dataset,labels,out_path,new_shape=(-1,28,28)):
    n_imgs=dataset.shape[0]
    for i in range(n_imgs):
        name_i=str(i)+'_'+str(labes[i]) +'.png'
        img_i=dataset[i].reshape(new_shape)
        out_i=out_path+'/'+name_i
        cv2.imwrite(out_i,img_i)

def save_object(nn,path):
    file_object = open(path,'wb')
    pickle.dump(nn,file_object)
    file_object.close()

def read_object(path):
    file_object = open(path,'r')
    obj=pickle.load(file_object)  
    file_object.close()
    return obj