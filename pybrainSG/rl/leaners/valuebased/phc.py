'''
Created on 2016/02/19

@author: takuya-hv2
'''
from pybrainSG.rl.leaners.valuebased.indexablevaluebased import IndexableValueBasedLearner
from scipy import zeros
from pybrain.utilities import r_argmax
import numpy as np
from pybrain.utilities import abstractMethod
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import one_to_n
from pybrain.structure.modules import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork
from scipy import r_, asarray
import copy

class PHC_FA(IndexableValueBasedLearner):
    """ 
    Policy hill climbing algorithm (with function approximation for Q-value and policy): 
    http://www.cs.cmu.edu/~mmv/papers/01ijcai-mike.pdf
    """
    
    learningRate = 0.5      # aka alpha: make sure this is being decreased by calls from the learning agent!
    learningRateDecay = 100 # aka n_0, but counting decay-calls
    
    randomInit = True
    
    rewardDiscount = 0.99 # aka gamma
    
    batchMode = False
    passNextAction = False # for the _updateWeights method
    
    def __init__(self, num_features, num_actions, indexOfAgent=None):
        IndexableValueBasedLearner.__init__(self, indexOfAgent)
        self.explorer = None        
        self.num_actions = num_actions
        self.num_features = num_features
        self.indexOfAgent=indexOfAgent
        self._behaviorPolicy = self._softmaxPolicy
        self.reset()
        self.ownerAgentProperties["requireOtherAgentsState"]=False
        self.ownerAgentProperties["requireJointAction"]=False
        self.ownerAgentProperties["requireJointReward"]=False
        
    def _pi(self, state):
        """ Return vector of probability of policy for all actions, 
        given the state(-features). """
        abstractMethod()
           
    def _softmaxPolicy(self, state):
        tmp = zeros(self.num_actions)
        pi=self._pi(state)
        rand=np.random.rand()
        cum=0.0
        for i in range(self.num_actions):
            cum+=pi[i]
            if rand < cum:
                tmp[i] = 1
                return tmp

    def reset(self):        
        IndexableValueBasedLearner.reset(self)        
        self._callcount = 0
        self.newEpisode()
    
    def newEpisode(self):  
        IndexableValueBasedLearner.newEpisode(self)      
    
    def _updateWeights(self, state, action, reward, next_state):
        '''
        Expected to update approximator. 
        '''
        abstractMethod()
        
        
class PHC_NN(PHC_FA):
    '''PHC with neural function approximation. '''
    delta=0.1
    maxNumberofAverage=30
    weightdecay=0.001
    trainingEpochPerUpdateWight=2
    
    def __init__(self, num_features, num_actions, indexOfAgent=None):    
        PHC_FA.__init__(self, num_features, num_actions, indexOfAgent)
        self.linQ = buildNetwork(num_features + num_actions, (num_features + num_actions), 1, hiddenclass = SigmoidLayer, outclass = LinearLayer)
        self.linPolicy = buildNetwork(num_features, (num_features + num_actions), num_actions, hiddenclass = SigmoidLayer,outclass = SigmoidLayer)
        self.trainer4LinQ=BackpropTrainer(self.linQ,weightdecay=self.weightdecay)
        self.trainer4LinPolicy=BackpropTrainer(self.linPolicy,weightdecay=self.weightdecay)

    def _pi(self, state):
        """Given state, compute probabilities for each action."""
        values = np.array(self.linPolicy.activate(r_[state]))
        z=np.sum(values)
        return (values/z).flatten()
    
    def _qValues(self, state):
        """ Return vector of q-values for all actions, 
        given the state(-features). """
        values = np.array([self.linQ.activate(r_[state, one_to_n(i, self.num_actions)]) for i in range(self.num_actions)])
        return values.flatten()

            
    def _updateWeights(self, state, action, reward, next_state):
        """ state and next_state are vectors, action is an integer. """
        #update Q-value function approximator
        target=reward + self.rewardDiscount * max(self._qValues(next_state))
        inp=r_[asarray(state), one_to_n(action, self.num_actions)]
        self.trainer4LinQ=BackpropTrainer(self.linQ,weightdecay=self.weightdecay)
        ds = SupervisedDataSet(self.num_features+self.num_actions,1)
        ds.addSample(inp, target)
        self.trainer4LinQ.trainOnDataset(ds)
        #Update policy
        bestAction=r_argmax(self._qValues(state))
        target= one_to_n(bestAction, self.num_actions)
        inp=r_[asarray(state)]
        ds = SupervisedDataSet(self.num_features,self.num_actions)
        ds.addSample(inp, target)
        self.trainer4LinPolicy=BackpropTrainer(self.linPolicy,
                                               learningrate=self.delta,
                                               weightdecay=self.weightdecay)
        self.trainer4LinPolicy.setData(ds)
        self.trainer4LinPolicy.trainEpochs(epochs=self.trainingEpochPerUpdateWight)
        




class PHC_WoLF_NN(PHC_FA):
    '''PHC_WoLF with neural function '''
    deltaW=0.05
    deltaL=0.2
    maxNumberofAverage=30
    weightdecay=0.001
    trainingEpochPerUpdateWight=1
    
    def __init__(self, num_features, num_actions, indexOfAgent=None):    
        PHC_FA.__init__(self, num_features, num_actions, indexOfAgent)
        self.linQ = buildNetwork(num_features + num_actions, (num_features + num_actions), 1, hiddenclass = SigmoidLayer, outclass = LinearLayer)
        self.linPolicy = buildNetwork(num_features, (num_features + num_actions), num_actions, hiddenclass = SigmoidLayer,outclass = SigmoidLayer)
        self.averagePolicy=[]
        self.trainer4LinQ=BackpropTrainer(self.linQ,weightdecay=self.weightdecay)
        self.trainer4LinPolicy=BackpropTrainer(self.linPolicy,weightdecay=self.weightdecay)

    def _pi(self, state):
        """Given state, compute softmax probability for each action."""
        values = np.array(self.linPolicy.activate(r_[state]))
        z=np.sum(values)
        return (values/z).flatten()
    
    def _qValues(self, state):
        """ Return vector of q-values for all actions, 
        given the state(-features). """
        values = np.array([self.linQ.activate(r_[state, one_to_n(i, self.num_actions)]) for i in range(self.num_actions)])
        return values.flatten()

    def _piAvr(self, state):
        pi=np.zeros(self.num_actions)
        for elem in self.averagePolicy:
            values = np.array(elem.activate(r_[state]))
            pi=np.add(pi.flatten(),values.flatten())
        z=np.sum(pi)
        pi=pi/z
        return pi.flatten()
        
    def _updateWeights(self, state, action, reward, next_state):
        """ state and next_state are vectors, action is an integer. """
        #update Q-value function approximator
        target=reward + self.rewardDiscount * max(self._qValues(next_state))
        inp=r_[asarray(state), one_to_n(action, self.num_actions)]
        self.trainer4LinQ=BackpropTrainer(self.linQ,weightdecay=self.weightdecay)
        ds = SupervisedDataSet(self.num_features+self.num_actions,1)
        ds.addSample(inp, target)        
        self.trainer4LinQ.trainOnDataset(ds)

        #update estimate of average policy
        self.averagePolicy.append(copy.deepcopy(self.linPolicy))
        if len(self.averagePolicy) > self.maxNumberofAverage:
            self.averagePolicy.pop(np.random.randint(len(self.averagePolicy)))
            
        #update policy function approximator
        delta=None
        cumRewardOfCurrentPolicy=0.0
        values=self._qValues(state)
        pi=self._pi(state)
        for elem_action in range(self.num_actions):
            cumRewardOfCurrentPolicy=pi[elem_action]*values[elem_action]
        cumRewardOfAveragePolicy=0.0
        api=self._piAvr(state)
        for elem_action in range(self.num_actions):
            cumRewardOfAveragePolicy=api[elem_action]*values[elem_action]
        if cumRewardOfCurrentPolicy > cumRewardOfAveragePolicy:
            delta=self.deltaW
        else:
            delta=self.deltaL
        
        #Update policy
        bestAction=r_argmax(self._qValues(state))
        target=one_to_n(bestAction, self.num_actions)
        inp=r_[asarray(state)]
        ds = SupervisedDataSet(self.num_features,self.num_actions)
        ds.addSample(inp, target)
        self.trainer4LinPolicy=BackpropTrainer(self.linPolicy,
                                               learningrate=(delta),
                                               weightdecay=self.weightdecay)
        self.trainer4LinPolicy.setData(ds)
        self.trainer4LinPolicy.trainEpochs(epochs=self.trainingEpochPerUpdateWight)
                        
        