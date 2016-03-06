'''
Created on 2016/02/19

@author: takuya-hv2
'''

from pybrainSG.rl.experiments.episodicSG import EpisodicExperimentSG
from pybrainSG.rl.leaners.valuebased.learnerfaSG import Q_LinFA_SG
from pybrainSG.rl.agents.linearfaSG import LinearFA_AgentSG
from pybrainSG.rl.examples.tasks.huntinggame import HuntingGame, HuntingGameTask
from pybrainSG.rl.agents.multiAgent import MultiAgent
import numpy as np
if __name__ == '__main__':
    ma=MultiAgent()
    for i in range(HuntingGame.numberofAgents):
        learner= Q_LinFA_SG(
                            num_features=(HuntingGame.numberofAgents*2+HuntingGame.numberofAnimals*2+1),
                            num_actions=len(HuntingGame.availableActions))
        agent= LinearFA_AgentSG(
                                learner,
                                num_features=np.ones(HuntingGame.numberofAgents)*(HuntingGame.numberofAgents*2+HuntingGame.numberofAnimals*2+1),
                                num_actions=np.ones(HuntingGame.numberofAgents)*len(HuntingGame.availableActions),
                                num_agents=HuntingGame.numberofAgents,
                                index=i)
        ma.addAgent(agent)
    task=HuntingGameTask()
    exp=EpisodicExperimentSG(task,ma)
    rewards=exp.doEpisodes(number=1000)
    print "Given reward for " + str(len(rewards)) + " episodes:"
    print "Reward for Agents"
    for i in range(len(rewards)):
        print str(rewards[i][-1][0])
    