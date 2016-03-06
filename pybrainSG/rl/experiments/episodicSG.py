'''
Created on 2016/02/19

@author: takuya-hv2
'''
__author__ = 'Takuya Hiraoka, takuya-h@is.naist.jp'

from pybrain.rl.experiments.experiment import Experiment
from pybrainSG.rl.agents.multiAgent import MultiAgent
from pybrainSG.rl.environments.episodicSG import EpisodicTaskSG

class EpisodicExperimentSG(Experiment):
    """ Stochastic version of EpisodicExperiment class. """
    def __init__(self, task, multiAgent):
        assert isinstance(task, EpisodicTaskSG), "task should be the subclass of EpisodicTaskSG."
        assert isinstance(multiAgent, MultiAgent), "task should be MultAgent."
        Experiment.__init__(self, task, multiAgent)


    def _oneInteraction(self):
        """ Do an interaction between the Task and Agents. """
        self.stepid += 1
        self.agent.integrateObservation(self.task.getObservation())
        self.task.performAction(self.agent.getJointAction())
        reward = self.task.getReward()
        self.agent.giveJointReward(reward)
        return reward

    def doEpisodes(self, number = 1):
        """ Do one episode, and return the joint rewards of each step as a list. """
        all_rewards = []
        for dummy in range(number):
            self.agent.newEpisode()
            rewards = []
            self.stepid = 0
            self.task.reset()
            while not self.task.isFinished():
                r = self._oneInteraction()
                rewards.append(r)
            all_rewards.append(rewards)
        return all_rewards
