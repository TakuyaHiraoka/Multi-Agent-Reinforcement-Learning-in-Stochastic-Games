'''
Created on 2016/02/20

@author: takuya-hv2
'''

from pybrain.rl.environments import Environment
from pybrainSG.rl.experiments.episodicSG import EpisodicTaskSG
import numpy as np


class StaticGameTask(EpisodicTaskSG):
    '''All agent make decision (Head or Tail) simultaneously only one time. '''

    isGameFinished=False
    def __init__(self):
        EpisodicTaskSG.__init__(self, SimpleMatrixGame())
    
    def reset(self):
        EpisodicTaskSG.reset(self)
        self.isGameFinished=False
    
    def isFinished(self):
        return self.isGameFinished

    def getReward(self):
        self.isGameFinished=True
        return self.env.getJointReward()


class SimpleMatrixGame(Environment):
    '''Corresponding to Heads and Tails respectively.'''
    availableActions=[0,1]
    '''payoff matrix of each agent in cooperative task scenario'''
    payoffMatricForAgent1=[[10,-10],
                            [-10,-10]]
    payoffMatricForAgent2=[[10,-10],
                            [-10,-10]]
#     '''payoff matrix of zero-sumgame scenario. nash equilibrium: (Agenat1's action=0,Agent2's action=1)'''
#     payoffMatricForAgent1=[[5,2],
#                            [-1,6]]
#     payoffMatricForAgent2=[[-5,-2],
#                            [1,-6]]
#     '''payoff matrix of zero-sumgame scenario. matching pennies'''
#     payoffMatricForAgent1=[[1,-1],
#                            [-1,1]]
#     payoffMatricForAgent2=[[-1,1],
#                            [1,-1]]
    
    outcomeForAfgenet1=None
    outcomeForAfgenet2=None
    
    def getSensors(self):
        return np.ones((2,1))#Static state (i.e., no state transition)
    
    def performAction(self, action):
        self.outcomeForAfgenet1=SimpleMatrixGame.payoffMatricForAgent1[action[0]][action[1]]
        self.outcomeForAfgenet2=SimpleMatrixGame.payoffMatricForAgent2[action[0]][action[1]]

    def reset(self):
        self.outcomeForAfgenet1=None
        self.outcomeForAfgenet2=None
        
    def getJointReward(self):
        return np.array([self.outcomeForAfgenet1,self.outcomeForAfgenet2])

