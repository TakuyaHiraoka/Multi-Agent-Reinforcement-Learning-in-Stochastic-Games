'''
Created on 2016/02/19

@author: takuya-hv2
'''

from pybrainSG.rl.experiments.episodicSG import EpisodicExperimentSG
from pybrainSG.rl.leaners.valuebased.ceq import *
from pybrainSG.rl.agents.ceqa import *
from pybrainSG.rl.examples.tasks.huntinggame import HuntingGame, HuntingGameTask
from pybrainSG.rl.agents.multiAgent import MultiAgent
#from pybrain.unsupervised.trainers.deepbelief

import numpy as np
if __name__ == '__main__':
#     warnings.simplefilter("ignore")
    ma=MultiAgent()
    HuntingGame.numberofAgents=2
    for i in range(HuntingGame.numberofAgents):
        learner= CEQ_Lin(
                            num_features=(HuntingGame.numberofAgents*2+HuntingGame.numberofAnimals*2+1),
                            num_actions=np.ones(HuntingGame.numberofAgents,dtype=np.int8)*len(HuntingGame.availableActions),
                            num_agents=HuntingGame.numberofAgents,
                            indexOfAgent=i)
        agent= CEQ_Agent(
                                learner,
                                num_features=np.ones(HuntingGame.numberofAgents,dtype=np.int8)*(HuntingGame.numberofAgents*2+HuntingGame.numberofAnimals*2+1),
                                num_actions=np.ones(HuntingGame.numberofAgents,dtype=np.int8)*len(HuntingGame.availableActions),
                                num_agents=HuntingGame.numberofAgents,
                                index=i)
        ma.addAgent(agent)
    task=HuntingGameTask()
    exp=EpisodicExperimentSG(task,ma)
    print "Reward for Agents"
    for i in range(40000):
        rewards=exp.doEpisodes(number=1)
        print str(rewards[0][-1][0])
    