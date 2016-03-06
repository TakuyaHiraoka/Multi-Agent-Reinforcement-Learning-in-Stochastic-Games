'''
Created on 2016/02/28

@author: takuya-hv2
'''
from pybrainSG.rl.agents.loggingSG import LoggingAgentSG
from pybrain.utilities import drawIndex
from pybrainSG.rl.leaners.valuebased.ceq import NFCEQ
from scipy import array

class NFCEQ_Agent(LoggingAgentSG):
    """
    Agent based on NFCEQ put on: 
    pybrainSG.rl.leaners.valuebased.ceq
    """    
    init_exploration = 0.3   # aka epsilon
    exploration_decay = 0.98 # per episode        
    
    init_temperature = 1.
    temperature_decay = 0.99 # per episode
    
    # flags for exploration strategies
    epsilonGreedy = True
    learning = True
    greedy = False
     
    def __init__(self, learner, num_features, num_actions, num_agents, index, **kwargs):
        LoggingAgentSG.__init__(self, num_features, num_actions, num_agents, index, **kwargs)
        assert isinstance(learner, NFCEQ), "learner should be instance of NFCEQ."
        self.learner = learner
        # if learner is available, tell it the module and data
        if self.learner is not None:
            self.learner.dataset = self.history
        self.learning = True
        self.learner._behaviorPolicy = self._actionProbs
        self.reset()

        self.agentProperties["requireOtherAgentsState"]=False
        self.agentProperties["requireJointAction"]=True
        self.agentProperties["requireJointReward"]=True
        for prop in self.learner.getProperty().keys():
            if learner.getProperty()[prop]:
                assert self.getProperty()[prop], "learners property should same to that of agents."
        
    def _actionProbs(self, state):
        if self.greedy:
            return self.learner._greedyPolicy(state)
        elif self.epsilonGreedy:
            return (self.learner._greedyPolicy(state) * (1 - self._expl_proportion) 
                    + self._expl_proportion / float(self.learner.num_actions[self.indexOfAgent]))
        else:
            return self.learner._boltzmannPolicy(state, self._temperature)                    
    
    def getAction(self):
        self.lastaction = drawIndex(self._actionProbs(self.lastobs), True)
        return array([self.lastaction])
        
    def integrateObservation(self, obs):
        LoggingAgentSG.integrateObservation(self, obs)        
        
    def reset(self):
        LoggingAgentSG.reset(self)
        self._temperature = self.init_temperature
        self._expl_proportion = self.init_exploration
        self.learner.reset()
        self.newEpisode()
        
    def newEpisode(self):
        """ Indicate the beginning of a new episode in the training cycle. """
        if self.logging:
            for i in range(self.numAgents):
                self.history[i].newSequence()
        if self.learning and not self.learner.batchMode:
            self.learner.newEpisode()
        else:
            self._temperature *= self.temperature_decay
            self._expl_proportion *= self.exploration_decay      
            self.learner.newEpisode()

            
    def learn(self,episodes):
        assert isinstance(self.learner,NFCEQ), "learner should be an instance of CEQ-NFQ"
        self.learner.learn()
    
    def setIndexOfAgent(self,index):
        """ set index to agent.
            :key index: index of agent
            :type index: integer
        """ 
        super(NFCEQ_Agent, self).setIndexOfAgent(index)
        self.learner.setIndexOfAgent(index)
        