'''
Created on 2016/02/19

@author: takuya-hv2
'''

from pybrainSG.rl.experiments.episodicSG import EpisodicExperimentSG
from pybrainSG.rl.leaners.valuebased.ceq import *
from pybrainSG.rl.agents.ceqa import *
from pybrainSG.rl.examples.tasks.staticgame import SimpleMatrixGame, StaticGameTask
from pybrainSG.rl.agents.multiAgent import MultiAgent
import numpy as np
if __name__ == '__main__':
    ma=MultiAgent()
    for i in range(2):
        
        learner= CEQ_Lin(
                            num_features=1,
                            num_actions=np.ones(2,dtype=np.int8)*len(SimpleMatrixGame.availableActions),
                            num_agents=2,
                            indexOfAgent=i)
        learner.rewardDiscount=0.0
        agent= CEQ_Agent(learner,
                            num_features=1,
                            num_actions=(np.ones(2)*len(SimpleMatrixGame.availableActions)),
                            num_agents=2,
                            index=i)
        ma.addAgent(agent)

    task=StaticGameTask()
    
    exp=EpisodicExperimentSG(task,ma)
    print "Reward for Agent 1, Reward for Agent 2"
    for i in range(50000):
        rewards=exp.doEpisodes(number=1)
        print str(rewards[0][-1][0])+", "+str(rewards[0][-1][1])
    