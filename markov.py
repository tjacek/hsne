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

    def __call__(self,theta,start_state):
        current_state=start_state #np.random.randint(self.n_states)
        for i in range(theta):
            j=self.next_state(current_state)
            current_state=self.states[current_state][j]
        return current_state

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
    return EffMarkovChain(np.array(trans),states)

def find_landmarks(markov_chain,beta=100,theta=50,beta_theshold=1.50):
    states=markov_chain.get_states()
    hist=np.zeros((len(states),))
    for state_i in states:
        print(state_i)
        for j in range(beta):
            end_state=markov_chain(theta,state_i)
            hist[end_state]+=1
    treshold=beta_theshold*beta
    landmarks=[ i 
                for i,hist_i in enumerate(hist)
                    if(hist_i>treshold)]
    return landmarks

def compute_influence(markov_chain,landmarks,beta=50):
    states=markov_chain.get_states()
    n_landmarks=len(landmarks)
    landmark_dict={ landmark_i:i 
                    for i,landmark_i in enumerate(landmarks)}
    landmarks=Set(landmarks)
    def influence_helper(state_i):
        print(state_i)
        influence=np.zeros((n_landmarks,))
        for j in range(beta):
            end_state=markov_chain.seek_landmark(state_i,landmarks)
            landmark_index=landmark_dict[end_state]
            influence[landmark_index]+=1
        influence/=float(beta)
        return influence
    return np.array([ influence_helper(state_i)  
                        for state_i in states])

def get_transition_matrix(new_landmark,old_landmarks):
    l1=len(new_landmark)
    l2=len(old_landmark)
    num=sum([I[k][i] * I[k][j] 
                for k in range(l1)])
    div=sum([sum([ I[k][i] * I[k][j]  
                    for k in range(l1) ]) 
                for l in range(l2)
            ])
    return num/div

if __name__ == "__main__": 
    make_markov_chain("mnist_graph")