'''
Created on 2016/02/19

@author: takuya-hv2
'''

from pybrainSG.rl.experiments.episodicSG import EpisodicExperimentSG
from pybrainSG.rl.leaners.valuebased.learnerfaSG import Q_LinFA_SG
from pybrainSG.rl.agents.linearfaSG import LinearFA_AgentSG
from pybrainSG.rl.examples.tasks.staticgame import SimpleMatrixGame, StaticGameTask
from pybrainSG.rl.agents.multiAgent import MultiAgent
import numpy as np
if __name__ == '__main__':
    ma=MultiAgent()
    for i in range(2):
        learner= Q_LinFA_SG(
                            num_features=1,
                            num_actions=len(SimpleMatrixGame.availableActions))
        agent= LinearFA_AgentSG(learner,
                            num_features=np.ones((2,1)),
                            num_actions=(np.ones(2)*len(SimpleMatrixGame.availableActions)),
                            num_agents=2,
                            index=i)
        ma.addAgent(agent)

    task=StaticGameTask()
    
    exp=EpisodicExperimentSG(task,ma)
    rewards=exp.doEpisodes(number=500)
    print "Given reward for " + str(len(rewards)) + " episodes:"
    print "Reward for Agent 1, Reward for Agent 2"
    for i in range(len(rewards)):
        print str(rewards[i][-1][0])+", "+str(rewards[i][-1][1])
    