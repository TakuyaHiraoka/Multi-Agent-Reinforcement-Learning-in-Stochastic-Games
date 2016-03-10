'''
Created on 2016/02/19

@author: takuya-hv2
'''
from pybrain.rl.learners.valuebased.interface import ActionValueNetwork
from pybrainSG.rl.experiments.episodicSG import EpisodicExperimentSG
from pybrainSG.rl.agents.multiAgent import MultiAgent
from pybrainSG.rl.leaners.valuebased.nfqSG import NFQ_SG
from pybrainSG.rl.agents.learningSG import LearningAgentSG
from pybrainSG.rl.examples.tasks.huntinggame import HuntingGame, HuntingGameTask
import numpy as np
import warnings
if __name__ == '__main__':
    warnings.simplefilter("ignore")
    for _ in range(500):
        ma=MultiAgent()
        #
        HuntingGame.numberofAgents=2
        for i in range(HuntingGame.numberofAgents):
            #dimState=# position of each agent in grid world + # position of each niman in grid world + bias
            net=ActionValueNetwork(dimState=(HuntingGame.numberofAgents*2+HuntingGame.numberofAnimals*2+1),
                                   numActions=len(HuntingGame.availableActions))
            learner= NFQ_SG(maxEpochs=100)#hopefully, more than 100.
#             learner._explorer.epsilon=0.1#In one player case, that too small. 
            #print learner.explorer
            agent = LearningAgentSG(net,
                                    num_features=(np.ones(HuntingGame.numberofAgents)*(HuntingGame.numberofAgents*2+HuntingGame.numberofAnimals*2+1)),
                                    num_actions=(np.ones(HuntingGame.numberofAgents)*len(HuntingGame.availableActions)), 
                                    num_agents=HuntingGame.numberofAgents, 
                                    learner=learner,
                                    index=i)
            ma.addAgent(agent)
        
        task=HuntingGameTask()
        
        exp=EpisodicExperimentSG(task,ma)
        print "Reward for Agents"
        print "Average Reward for Agents (at the end of episode)"
        #Two phase leanring
        rewards=exp.doEpisodes(number=10)#firstphase
        ma.learn()
        for numBatch in range(40):
            avr=0.0
            for i in range(len(rewards)):
                #print str(rewards[i][-1][0])
                #average
                avr+=rewards[i][-1][0]
            avr/=float(len(rewards))
            print avr
            rewards=exp.doEpisodes(number=10)
            ma.learn()
