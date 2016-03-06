'''
Created on 2016/02/19

@author: takuya-hv2
'''
from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner

class IndexableValueBasedLearner(ValueBasedLearner):
    indexOfAgent=None
    ownerAgentProperties={
                     "requireOtherAgentsState": None, #Define if learner require, in addition to owner's state information, other agent state information as well. 
                     "requireJointAction":None, #Define if learner require, in addition to owner's state information, other agent action information as well.   
                     "requireJointReward":None}#Define if learner require, in addition to owner's state information, other agent reward information as well. 

    def __init__(self, indexOfAgent=None, **kwargs):
        ValueBasedLearner.__init__(self)
        self.indexOfAgent=indexOfAgent
    
    def setIndexOfAgent(self, indexOfAgent):
        self.indexOfAgent=indexOfAgent

    def getProperty(self):
        for elem in self.ownerAgentProperties.values():
            assert isinstance(elem,bool), "All property should be initialize with boolian."
        return self.ownerAgentProperties