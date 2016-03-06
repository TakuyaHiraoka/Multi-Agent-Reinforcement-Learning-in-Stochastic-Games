'''
Created on 2016/02/19

@author: takuya-hv2
'''
from pybrain.utilities import abstractMethod
from pybrain.rl.environments.task import Task

class EpisodicTaskSG(Task):
    """Stochastic game version of EpisodicTask class"""
    
    def __init__(self, environment):
        Task.__init__(self,environment)
    
    def reset(self):
        """ Re-initialize the environment """
        self.env.reset()

    def isFinished(self):
        """ Is the current episode over? """
        abstractMethod()

    def performAction(self, jointAction):
        """ Execute joint action of all agents. """
        Task.performAction(self, jointAction)
