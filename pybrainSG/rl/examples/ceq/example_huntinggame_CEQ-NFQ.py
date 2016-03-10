'''
Created on 2016/02/28

@author: takuya-hv2
'''
from pybrainSG.rl.experiments.episodicSG import EpisodicExperimentSG
from pybrainSG.rl.leaners.valuebased.ceq import *
from pybrainSG.rl.agents.nfceqa import *
from pybrainSG.rl.examples.tasks.huntinggame import HuntingGame, HuntingGameTask
from pybrainSG.rl.agents.multiAgent import MultiAgent

import numpy as np
if __name__ == '__main__':
    warnings.simplefilter("ignore")
    
    for _ in range(500):
        ma=MultiAgent()
        HuntingGame.numberofAgents=2
        for i in range(HuntingGame.numberofAgents):
            learner= NFCEQ(
                                num_features=(HuntingGame.numberofAgents*2+HuntingGame.numberofAnimals*2+1),
                                num_actions=np.ones(HuntingGame.numberofAgents,dtype=np.int8)*len(HuntingGame.availableActions),
                                num_agents=HuntingGame.numberofAgents,
                                max_epochs=100,
                                indexOfAgent=i)
            agent= NFCEQ_Agent(
                                    learner,
                                    num_features=np.ones(HuntingGame.numberofAgents,dtype=np.int8)*(HuntingGame.numberofAgents*2+HuntingGame.numberofAnimals*2+1),
                                    num_actions=np.ones(HuntingGame.numberofAgents,dtype=np.int8)*len(HuntingGame.availableActions),
                                    num_agents=HuntingGame.numberofAgents,
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
