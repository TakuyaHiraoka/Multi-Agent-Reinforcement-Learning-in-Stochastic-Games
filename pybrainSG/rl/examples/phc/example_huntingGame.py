'''
Created on 2016/02/19

@author: takuya-hv2
'''
from pybrainSG.rl.experiments.episodicSG import EpisodicExperimentSG
from pybrainSG.rl.examples.tasks.huntinggame import HuntingGame, HuntingGameTask
from pybrainSG.rl.agents.multiAgent import MultiAgent
from pybrainSG.rl.leaners.valuebased.phc import *
from pybrainSG.rl.agents.faphc import PHC_Agent

if __name__ == '__main__':
    ma=MultiAgent()
    HuntingGame.numberofAgents=2
    for i in range(HuntingGame.numberofAgents):
#         learner= PHC_NN(
#                         num_features=(HuntingGame.numberofAgents*2+HuntingGame.numberofAnimals*2+1),
#                         num_actions=len(HuntingGame.availableActions))
        learner= PHC_WoLF_NN(
                             num_features=(HuntingGame.numberofAgents*2+HuntingGame.numberofAnimals*2+1),
                             num_actions=len(HuntingGame.availableActions))
        agent= PHC_Agent(learner,numAgents=HuntingGame.numberofAgents,index=i)
        ma.addAgent(agent)
    
    task=HuntingGameTask()
    
    print "Given reward for Agents"
    for i in range(10000):
        exp=EpisodicExperimentSG(task,ma)
        rewards=exp.doEpisodes(number=1)
        for i in range(len(rewards)):
            print str(rewards[i][-1][0])
        
    