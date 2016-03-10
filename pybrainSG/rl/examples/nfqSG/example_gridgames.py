'''
Created on 2016/03/06

@author: takuya-hv2
'''
from pybrain.rl.learners.valuebased.interface import ActionValueNetwork
from pybrainSG.rl.experiments.episodicSG import EpisodicExperimentSG
from pybrainSG.rl.agents.multiAgent import MultiAgent
from pybrainSG.rl.leaners.valuebased.nfqSG import NFQ_SG
from pybrainSG.rl.agents.learningSG import LearningAgentSG
from pybrainSG.rl.examples.tasks.gridgames import GridGameTask, GridGame
import numpy as np
import warnings

if __name__ == '__main__':
    warnings.simplefilter("ignore")
    for _ in range(500):
        ma=MultiAgent()
        for i in range(GridGame.numberofAgents):
            net=ActionValueNetwork(dimState=(GridGame.numberofAgents*2),
                                   numActions=len(GridGame.availableActions))
            learner= NFQ_SG(maxEpochs=100)
            agent = LearningAgentSG(net,
                                    num_features=(np.ones(GridGame.numberofAgents)*(GridGame.numberofAgents*2)),
                                    num_actions=(np.ones(GridGame.numberofAgents)*len(GridGame.availableActions)), 
                                    num_agents=GridGame.numberofAgents, 
                                    learner=learner,
                                    index=i)
            ma.addAgent(agent)
#         task=GridGameTask(gameType="GG1")
#         task=GridGameTask(gameType="GG2")
        task=GridGameTask(gameType="GG3")
        exp=EpisodicExperimentSG(task,ma)
        print "Average reward for agents at the end of episode:"
        #Two phase learning
        rewards=exp.doEpisodes(number=10)#first phase
        ma.learn()
        for numBatch in range(40):
            avr=np.array([0.0,0.0])
            for i in range(len(rewards)):
                avr+=rewards[i][-1]
            avr/=float(np.size(rewards,axis=0))
            print avr
            rewards=exp.doEpisodes(number=10)
            ma.learn()
