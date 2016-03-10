'''
Created on 2016/03/09

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

class ON_SGSP_FA(IndexableValueBasedLearner):
    """ 
    Stochastic game sub-problem (with function approximation for Q-value and policy): 
    http://www.ifaamas.org/Proceedings/aamas2015/aamas/p1371.pdf
    """
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
        self.ownerAgentProperties["requireJointAction"]=True
        self.ownerAgentProperties["requireJointReward"]=True
        
    def _pi(self, state):
        """ Return vector of probability of policy for all actions, 
        given the state(-features). """
        abstractMethod()
           
    def _softmaxPolicy(self, state):
        tmp = zeros(self.num_actions[self.indexOfAgent])
        pi=self._pi(state)
        rand=np.random.rand()
        cum=0.0
        for i in range(self.num_actions[self.indexOfAgent]):
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
        
        
class ON_SGSP_NN(ON_SGSP_FA):
    '''ON_SGSP with neural function approximation. '''
    weightdecay=0.01
    zeta=0.00001
    #
    cn=0.05
    bn=0.05
    #
    decayCn=0.9999
    decayBn=0.9995

    
    def __init__(self, num_features, num_actions, num_agents, index):    
        ON_SGSP_FA.__init__(self, num_features, num_actions, index)
        self.num_agents= num_agents
        self.linQ = []
        for iAgent in range(self.num_agents):
            self.linQ.append(buildNetwork(num_features + num_actions[iAgent], 
                                         num_features*2, 
                                         1, 
                                         hiddenclass = SigmoidLayer, 
                                         outclass = LinearLayer))
        self.linGradient = buildNetwork(num_features + num_actions[self.indexOfAgent], 
                                     (num_features + num_actions[self.indexOfAgent])*2, 
                                     1, 
                                     hiddenclass = SigmoidLayer, 
                                     outclass = LinearLayer)
        self.linPolicy = buildNetwork(num_features, 
                                       (num_features + num_actions[self.indexOfAgent])*2, 
                                       num_actions[self.indexOfAgent], 
                                       hiddenclass = SigmoidLayer, 
                                       outclass = SigmoidLayer)
        assert self.decayBn < self.decayCn, "Cn shold be bigger than Bn."
            
    def _pi(self, state):
        """Given state, compute probabilities for each action."""
        values = np.array(self.linPolicy.activate(r_[state]))
        z=np.sum(values)
        return (values/z).flatten()

    def _qValues(self, state, iAgent):
        """ Return vector of q-values for all actions, 
        given the state(-features). """
        values = np.array([self.linQ[iAgent].activate(r_[state, one_to_n(i, self.num_actions[iAgent])]) for i in range(self.num_actions[iAgent])])
        return values.flatten()
    
    def _sgn(self, val):
        if val > self.zeta:
            return 1.0
        elif val < (-1.0*self.zeta):
            return -1.0
        else:
            return val
    
    def _gamma(self, val):
        if val > 1.0:
            return 1.0
        elif val < 0:
            return 0.0
        else:
            return val
        
                    
    def _updateWeights(self, state, action, reward, next_state): 
        """ state and next_state are vectors, action is an integer. """
        #update Q-value function approximator (estimate Q-value instead of V) 
        BellmanErrors=np.zeros(self.num_agents)
        for iAgent in range(self.num_agents):
            vValC=self._qValues(state,iAgent)
            vValN=self._qValues(next_state,iAgent)
            vArgMaxValC=r_argmax(vValC)
            vArgMaxValN=r_argmax(vValN)
            BellmanError=(reward[iAgent] + self.rewardDiscount * vValN[vArgMaxValN]) - vValC[vArgMaxValC]
            target=vValC[action[iAgent]]+self.cn*((reward[iAgent] + self.rewardDiscount * vValN[vArgMaxValN]) - vValC[action[iAgent]])
            BellmanErrors[iAgent]=BellmanError
            inp=r_[state, one_to_n(action[iAgent], self.num_actions[iAgent])]
            ds = SupervisedDataSet(self.num_features+self.num_actions[iAgent],1)
            ds.addSample(inp, target)
            BackpropTrainer(self.linQ[iAgent], learningrate=1.0, weightdecay=self.weightdecay).trainOnDataset(ds)
            
        #Estimate gradient
        grad=self.linGradient.activate(np.r_[asarray(state), one_to_n(action[self.indexOfAgent], self.num_actions[self.indexOfAgent])])[0]
        target=grad+self.cn*(np.sum(BellmanErrors, axis=0)-grad)
        inp=np.r_[asarray(state), one_to_n(action[self.indexOfAgent], self.num_actions[self.indexOfAgent])]
        ds = SupervisedDataSet(self.num_features+self.num_actions[self.indexOfAgent],1)
        ds.addSample(inp, target)
        BackpropTrainer(self.linGradient, learningrate=1.0,weightdecay=self.weightdecay).trainOnDataset(ds)
#         print str(self.indexOfAgent) + "-th agents optimization info.:"
#         print "All Bellman errors: "+str(np.sum(BellmanErrors, axis=0))
#         print "Self Bellman error: " + str(np.absolute(BellmanErrors[self.indexOfAgent]))
#         print "Self Q-value: " + str(self._qValues(state,self.indexOfAgent))
        #Update policy
        c_pi=self._pi(state)
#         print "Policy: " + str(c_pi)
        firstTerm=c_pi[action[self.indexOfAgent]]
        secondTerm=(np.sqrt(firstTerm) 
                    * np.absolute(BellmanErrors[self.indexOfAgent]) 
                    * self._sgn(-1.0*self.linGradient.activate(np.r_[asarray(state), one_to_n(action[self.indexOfAgent], self.num_actions[self.indexOfAgent])])[0])
                    )
        target=c_pi
        target[action[self.indexOfAgent]]=self._gamma(firstTerm - self.bn * secondTerm)
        inp=r_[asarray(state)]
        ds = SupervisedDataSet(self.num_features, self.num_actions[self.indexOfAgent])
        ds.addSample(inp, target)
        BackpropTrainer(self.linPolicy, learningrate=1.0,weightdecay=self.weightdecay).trainOnDataset(ds)
        
        #update bn, cn
        self.bn = self.bn * self.decayBn
        self.cn = self.cn * self.decayCn
        
        
        
        
