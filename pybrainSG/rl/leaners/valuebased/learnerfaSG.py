'''
Created on 2016/02/19

@author: takuya-hv2
'''
from pybrainSG.rl.leaners.valuebased.indexablevaluebased import IndexableValueBasedLearner
from scipy import zeros, dot, exp, clip, randn
from pybrain.utilities import r_argmax, setAllArgs

class LinearFALearnerSG(IndexableValueBasedLearner):
    """
    Stochastic game version of LinearFALearner
    """
    
    learningRate = 0.5      # aka alpha: make sure this is being decreased by calls from the learning agent!
    learningRateDecay = 100 # aka n_0, but counting decay-calls
    
    randomInit = True
    
    rewardDiscount = 0.99 # aka gamma
    
    batchMode = False
    passNextAction = False # for the _updateWeights method   
    #
    
    
    def __init__(self, num_features, num_actions, indexOfAgent=None, **kwargs):
        IndexableValueBasedLearner.__init__(self, indexOfAgent)
        setAllArgs(self, kwargs)
        self.explorer = None        
        self.indexOfAgent=indexOfAgent
        self.num_actions = num_actions
        self.num_features = num_features
        if self.randomInit:
            self._theta = randn(self.num_actions, self.num_features) / 10.
        else:
            self._theta = zeros((self.num_actions, self.num_features))
        self._additionalInit()
        self._behaviorPolicy = self._boltzmannPolicy
        self.reset()
        #
        self.ownerAgentProperties["requireOtherAgentsState"]=False
        self.ownerAgentProperties["requireJointAction"]=False
        self.ownerAgentProperties["requireJointReward"]=False

        
    def _additionalInit(self):
        pass
        
    def _qValues(self, state):
        """ Return vector of q-values for all actions, 
        given the state(-features). """
        return dot(self._theta, state)
    
    def _greedyAction(self, state):
        return r_argmax(self._qValues(state))
    
    def _greedyPolicy(self, state):
        tmp = zeros(self.num_actions)
        tmp[self._greedyAction(state)] = 1
        return tmp
    
    def _boltzmannPolicy(self, state, temperature=1.):
        tmp = self._qValues(state)
        return LinearFALearnerSG._boltzmannProbs(tmp, temperature)
    
    @staticmethod
    def _boltzmannProbs(qvalues, temperature=1.):
        if temperature == 0:
            tmp = zeros(len(qvalues))        
            tmp[r_argmax(qvalues)] = 1.
        else:
            tmp = qvalues / temperature            
            tmp -= max(tmp)        
            tmp = exp(clip(tmp, -20, 0))
        return tmp / sum(tmp)

    def reset(self):        
        IndexableValueBasedLearner.reset(self)        
        self._callcount = 0
        self.newEpisode()
    
    def newEpisode(self):  
        IndexableValueBasedLearner.newEpisode(self)      
        self._callcount += 1
        self.learningRate *= ((self.learningRateDecay + self._callcount) 
                              / (self.learningRateDecay + self._callcount + 1.))
    
    
class Q_LinFA_SG(LinearFALearnerSG):
    """ Standard Q-learning with linear FA. """
    
    def _updateWeights(self, state, action, reward, next_state):
        """ state and next_state are vectors, action is an integer. """
        td_error = reward + self.rewardDiscount * max(dot(self._theta, next_state)) - dot(self._theta[action], state) 
        #print(action, reward, td_error,self._theta[action], state, dot(self._theta[action], state))
        #print(self.learningRate * td_error * state)
        #print()
        self._theta[action] += self.learningRate * td_error * state 
        
