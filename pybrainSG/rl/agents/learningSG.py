'''
Created on 2016/02/19

@author: takuya-hv2
'''
from pybrainSG.rl.agents.loggingSG import LoggingAgentSG
from pybrainSG.rl.leaners.valuebased.indexablevaluebased import IndexableValueBasedLearner

class LearningAgentSG(LoggingAgentSG):
    """ 
    Variation of LearningAgent (pybrain.rl.agents.learning) for stochastic game, 
    which can use some single-agent reinforcement learnings (currently only NFQ) put on: 
    pybrainSG.rl.leaners.valuebased.nfqSG
    """

    def __init__(self, module, num_features, num_actions, num_agents, index, learner):
        """
        :key module: the acting module
        :key learner: the learner (optional) """
        assert isinstance(learner, IndexableValueBasedLearner), "learner should be indexable."
        self.module = module
        self.learner = learner
        LoggingAgentSG.__init__(self, num_features, num_actions, num_agents,index)

        # if learner is available, tell it the module and data
        if self.learner is not None:
            self.learner.module = self.module
            self.learner.dataset = self.history

        self.learning = True
        
        self.agentProperties["requireOtherAgentsState"]=False
        self.agentProperties["requireJointAction"]=False
        self.agentProperties["requireJointReward"]=False
        #parity check
        for prop in self.learner.getProperty().keys():
            if learner.getProperty()[prop]:
                assert self.getProperty()[prop], "learners property should same to that of agents."

    def _getLearning(self):
        """ Return whether the agent currently learns from experience or not. """
        return self.__learning


    def _setLearning(self, flag):
        """ Set whether or not the agent should learn from its experience """
        if self.learner is not None:
            self.__learning = flag
        else:
            self.__learning = False

    learning = property(_getLearning, _setLearning)


    def getAction(self):
        """ Activate the module with the last observation, add the exploration from
            the explorer object and store the result as last action. """
        LoggingAgentSG.getAction(self)

        self.lastaction = self.module.activate(self.lastobs)

        if self.learning:
            self.lastaction = self.learner.explore(self.lastobs, self.lastaction)

        return self.lastaction


    def newEpisode(self):
        """ Indicate the beginning of a new episode in the training cycle. """
        # reset the module when a new episode starts.
        self.module.reset()
        
        if self.logging:
            for i in range(self.numAgents):
                self.history[i].newSequence()

        # inform learner about the start of a new episode
        if self.learning:
            self.learner.newEpisode()

    def reset(self):
        """ Clear the history of the agent and resets the module and learner. """
        LoggingAgentSG.reset(self)
        self.module.reset()
        if self.learning:
            self.learner.reset()


    def learn(self, episodes=1):
        """ Call the learner's learn method, which has access to both module and history. """
        if self.learning:
            self.learner.learnEpisodes(episodes)

    def setIndexOfAgent(self,index):
        """ set index to agent.
            :key index: index of agent
            :type index: integer
        """ 
        super(LearningAgentSG, self).setIndexOfAgent(index)
        self.learner.setIndexOfAgent(index)

