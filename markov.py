import numpy as np
import knn,utils

class FiniteMarkovChain(object):
    def __init__(self, distributions):	
        self.distributions=distributions
        self.state=None

    def __call__(self,theta):
        self.state=np.random.randint(len(self.distributions))
        for i in range(theta):
            self.state=self.distributions[self.state]()
        return self.state

class FiniteDistribution(object):
    def __init__(self,states,probs):
        self.states=states
        self.probs=probs

    def __call__(self):
        print(sum(self.probs))
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



if __name__ == "__main__": 
    make_markov_chain("mnist_graph")