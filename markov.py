import numpy as np
import knn,utils

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
            self.state=self.distributions[self.state]()
        return self.state
    
    def get_states(self):
        return range(len(self.distributions))

class FiniteDistribution(object):
    def __init__(self,states,probs):
        self.states=states
        self.probs=probs

    def __call__(self):
        return np.random.choice(self.states, 1, p=self.probs)	
		
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

if __name__ == "__main__": 
    make_markov_chain("mnist_graph")