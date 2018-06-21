import numpy as np
import knn,utils
from sets import Set
import random

class EffMarkovChain(object):
    def __init__(self, trans,states):
        self.trans=trans
        self.states=states
        self.n_states=trans.shape[0]
        self.k=trans.shape[1]

    def get_states(self):
        return range(self.n_states)

    def __call__(self,beta,theta,start_state):
        result=np.zeros((beta,))
        for s in xrange(beta):
            current_state=start_state 
            for t in xrange(theta):
                i=self.next_state(current_state)
                result[s]=self.states[current_state][i]
        return result

    def next_state(self,state_i):
        r=random.random()
        for j in xrange(self.k):
            if(r<self.trans[state_i][j]):
                return j
        return self.k

    def seek_landmark(self,start,landmarks):
        current_state=start
        while(not (current_state in landmarks)):
            j=self.next_state(current_state)
            current_state=self.states[current_state][j]
        return current_state

    def save(self,trans_file='trans.txt',states_file='states.txt'):
        utils.save_array(self.trans,trans_file) 
        utils.save_array(self.states,states_file,prec='%i')    
                  
def make_eff_markov_chain(nn_graph):
    trans=[]
    states=[]
    for i in range(len(nn_graph)):
        names_i,distances_i=nn_graph[i]
        sigma_i=np.min(distances_i[distances_i!=0]) 
        dist_i=np.exp(distances_i/sigma_i)
        dist_i/=np.sum(dist_i)
        dist_i=np.cumsum(dist_i)
        trans.append(dist_i)
        states.append(names_i)
    return EffMarkovChain(np.array(trans),np.array(states))

def find_landmarks(markov_chain,beta=100,theta=50,beta_theshold=3.0):
    states=markov_chain.get_states()
    hist=np.zeros((len(states),))
    for state_i in states:
        if( state_i % 10 ==0):
            print(state_i)
        end_states=markov_chain(beta,theta,state_i)
        for end_state_i in end_states:
            hist[end_state_i]+=1
    treshold=beta_theshold*beta
    landmarks=[ i 
                for i,hist_i in enumerate(hist)
                    if(hist_i>treshold)]
    return landmarks

def compute_influence(markov_chain,landmarks,beta=50):  
    n_states=len(markov_chain.get_states())
    n_landmarks=len(landmarks)
    infl_matrix=np.zeros((n_states,n_landmarks),dtype=float)
    landmark_dict={ landmark_i:i 
                    for i,landmark_i in enumerate(landmarks)}
    landmarks=Set(landmarks)
    for state_i in range(n_states):
        print(state_i)
        for j in range(beta):
            end_state=markov_chain.seek_landmark(state_i,landmarks)
            landmark_index=landmark_dict[end_state]
            infl_matrix[state_i][landmark_index]+=1.0
    infl_matrix/=float(beta)
    return infl_matrix

def get_prob_matrix(infl_matrix,W):
    weighted_infl=infl_matrix.multiply(W)
    sp=weighted_infl.transpose()*infl_matrix
    T=sp.toarray()
    print(T.shape)
    return T

def to_cum_matrix(matrix):
    const=np.sum(matrix,axis=1)
    const=1.0/const
    print("T")
    n_dist=const.shape[0]
    prob=np.array([const[i]* row_i 
                    for i,row_i in enumerate(matrix)])
    prob=np.cumsum(prob,axis=1)
    return prob
    
if __name__ == "__main__": 
    make_markov_chain("mnist_graph")