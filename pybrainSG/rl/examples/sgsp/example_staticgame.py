'''
Created on 2016/03/10

@author: takuya-hv2
'''

from pybrainSG.rl.experiments.episodicSG import EpisodicExperimentSG
from pybrainSG.rl.agents.multiAgent import MultiAgent
from pybrainSG.rl.leaners.valuebased.sgsp import *
from pybrainSG.rl.agents.sgspa import SGSP_Agent
from pybrainSG.rl.examples.tasks.staticgame import SimpleMatrixGame, StaticGameTask

if __name__ == '__main__':
    ma=MultiAgent()
    for i in range(2):
        learner= ON_SGSP_NN(
                             num_features=1, 
                             num_actions=np.ones(2,dtype=np.int8)*len(SimpleMatrixGame.availableActions),
                             num_agents=2,
                             index=i)
        learner.rewardDiscount=0.0
        agent= SGSP_Agent(
                          learner,
                          num_actions=np.ones(2,dtype=np.int8)*len(SimpleMatrixGame.availableActions),
                          numAgents=2,
                          index=i)
        ma.addAgent(agent)
    task=StaticGameTask()
    
    exp=EpisodicExperimentSG(task,ma)
    rewards=exp.doEpisodes(number=1000)
    print "Given reward for " + str(len(rewards)) + " episodes:"
    print "Reward for Agent 1, Reward for Agent 2"
    for i in range(len(rewards)):
        print str(rewards[i][-1][0])+", "+str(rewards[i][-1][1])
    