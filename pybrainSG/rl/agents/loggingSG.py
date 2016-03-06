'''
Created on 2016/02/19

@author: takuya-hv2
'''
from pybrainSG.rl.agents.indexable import IndexableAgent
from pybrain.datasets.reinforcement import ReinforcementDataSet
import numpy as np
class LoggingAgentSG(IndexableAgent):
    """ This agent stores actions, states, and rewards encountered during
        interaction with an environment in a ReinforcementDataSet (which is
        a variation of SequentialDataSet).
        The stored history can be used for learning and is erased by resetting
        the agent. It also makes sure that integrateObservation, getAction and
        giveReward are called in exactly that order.
    """

    logging = True

    lastobs = None
    lastaction = None
    lastreward = None

    agentProperties={
                     "requireOtherAgentsState": None, #Define if agent require other agent state information. 
                     "requireJointAction":None, #Define if agent require other agent action information.  
                     "requireJointReward":None}#Define if agent require other agent reward information.


    def __init__(self, indims, outdims, numAgents, index=None, **kwargs):
        IndexableAgent.__init__(self, index)
        self.setArgs(**kwargs)
        
        # store input and output dimension #input, output dimension for each agent 
        self.indim = indims
        self.outdim = outdims
        self.numAgents=numAgents
        # create the history dataset
        self.history=[]
        for i in range(self.numAgents):
            self.history.append(ReinforcementDataSet(self.indim[i], self.outdim[i]))


    def integrateObservation(self, obs):
        """Step 1: store the observation received in a temporary variable until action is called and
        reward is given. """
        self.lastobs = obs
        self.lastaction = None
        self.lastreward = None


    def getAction(self):
        """Step 2: store the action in a temporary variable until reward is given. """
        assert self.lastobs != None
        assert self.lastaction == None
        assert self.lastreward == None
        # implement getAction in subclass and set self.lastaction


    def giveReward(self, r):
        """Step 3: store observation, action and reward in the history dataset. """
        # step 3: assume that state and action have been set
        assert self.lastobs != None
        assert self.lastaction != None
        assert self.lastreward == None

        self.lastreward = r

        # store state, action and reward in dataset if logging is enabled
        if self.logging:
            for i in range(self.numAgents):
                tlastobs=None
                tlastaction=None
                tlastreward=None
                
                if self.getProperty()["requireOtherAgentsState"]:
                    tlastobs=self.lastobs[i]
                elif i==self.indexOfAgent:
                    tlastobs=self.lastobs
                else:
                    tlastobs=np.zeros(self.indim[i])
                if self.getProperty()["requireJointAction"]:
                    tlastaction=self.lastaction[i]
                elif i==self.indexOfAgent:
                    tlastaction=self.lastaction
                else:
                    tlastaction=np.zeros(self.outdim[i])
                if self.getProperty()["requireJointReward"]:
                    tlastreward=self.lastreward[i]
                elif i==self.indexOfAgent:
                    tlastreward=self.lastreward
                else:
                    tlastreward=np.zeros(1)
                self.history[i].addSample(tlastobs, tlastaction, tlastreward)
                     
    def newEpisode(self):
        """ Indicate the beginning of a new episode in the training cycle. """
        if self.logging:
            for i in range(self.numAgents):
                self.history[i].newSequence()


    def reset(self):
        """ Clear the history of the agent. """
        self.lastobs = None
        self.lastaction = None
        self.lastreward = None
        for i in range(self.numAgents):
            self.history[i].clear()

    def getProperty(self):
        for elem in self.agentProperties.values():
            assert isinstance(elem,bool), "All property should be initialize with proper boolean values."
        return self.agentProperties
    