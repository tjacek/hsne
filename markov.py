import numpy as np
import knn,utils
from sets import Set

class FiniteMarkovChain(object):
    def __init__(self, distributions):	
        self.distributions=distributions
        self.state=None

    def __call__(self,theta,start=None):
        if(start is None):
            self.state=np.random.randint(len(self.distributions))
        else:
            self.state=start
        for i in range(theta):
            self.next_state()
        return self.state
    
    def next_state(self):
        self.state=self.distributions[self.state]()

    def get_states(self):
        return range(len(self.distributions))

    def seek_landmark(self,start,landmarks):
        self.state=start
        while(not (self.state in landmarks)):
            self.next_state()
        return self.state    

class FiniteDistribution(object):
    def __init__(self,states,probs):
        self.states=states
        self.probs=probs

    def __call__(self):
        return np.random.choice(self.states, 1, p=self.probs)[0]	
		
def make_markov_chain(nn_graph):
    def dist_helper(i):
        names_i,distances_i=nn_graph[i]
        sigma_i=np.min(distances_i[distances_i!=0]) 
        dist=np.exp(distances_i/sigma_i)
        dist/=np.sum(dist)
        return FiniteDistribution(names_i,dist)
    dists=[dist_helper(i)
            for i in range(len(nn_graph))]
    return FiniteMarkovChain(dists)

def find_landmarks(markov_chain,beta=100,theta=50,beta_theshold=1.30):
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