'''
Created on 2016/03/06

@author: takuya-hv2
'''
from pybrainSG.rl.experiments.episodicSG import EpisodicExperimentSG
from pybrainSG.rl.agents.multiAgent import MultiAgent
from pybrainSG.rl.leaners.valuebased.ceq import *
from pybrainSG.rl.agents.nfceqa import *
from pybrainSG.rl.examples.tasks.gridgames import GridGameTask, GridGame
import numpy as np
import warnings

if __name__ == '__main__':
    warnings.simplefilter("ignore")
    for _ in range(500):
        ma=MultiAgent()
        for i in range(GridGame.numberofAgents):
            learner= NFCEQ(
                            num_features=(GridGame.numberofAgents*2),
                            num_actions=np.ones(GridGame.numberofAgents,dtype=np.int8)*len(GridGame.availableActions),
                            num_agents=GridGame.numberofAgents,
                            max_epochs=100,
                            indexOfAgent=i)
            agent= NFCEQ_Agent(
                                learner,
                                num_features=np.ones(GridGame.numberofAgents,dtype=np.int8)*(GridGame.numberofAgents*2),
                                num_actions=np.ones(GridGame.numberofAgents,dtype=np.int8)*len(GridGame.availableActions),
                                num_agents=GridGame.numberofAgents,
                                index=i)
            ma.addAgent(agent)
#         task=GridGameTask(gameType="GG1")
#         task=GridGameTask(gameType="GG2")
        task=GridGameTask(gameType="GG3")
        exp=EpisodicExperimentSG(task,ma)
        print "Average reward for agents at the end of episode:"
        #Two phase learning
        rewards=exp.doEpisodes(number=30)#first phase
        ma.learn()
        for numBatch in range(40):
            avr=np.array([0.0,0.0])
            for i in range(len(rewards)):
                avr+=rewards[i][-1]
            avr/=float(np.size(rewards,axis=0))
            print avr
            rewards=exp.doEpisodes(number=10)
            ma.learn()
