'''
Created on 2016/03/07

@author: takuya-hv2
'''
from pybrainSG.rl.experiments.episodicSG import EpisodicExperimentSG
from pybrainSG.rl.examples.tasks.gridgames import *
from pybrainSG.rl.agents.multiAgent import MultiAgent
from pybrainSG.rl.leaners.valuebased.learnerfaSG import Q_LinFA_SG
from pybrainSG.rl.agents.linearfaSG import LinearFA_AgentSG

if __name__ == '__main__':
    ma=MultiAgent()
    for i in range(GridGame.numberofAgents):
        learner= Q_LinFA_SG(
                         num_features=(GridGame.numberofAgents*2),
                         num_actions=len(GridGame.availableActions))
        agent= LinearFA_AgentSG(
                                learner,
                                num_features=np.ones(GridGame.numberofAgents,dtype=np.int8)*(GridGame.numberofAgents*2),
                                num_actions=np.ones(GridGame.numberofAgents,dtype=np.int8)*len(GridGame.availableActions),
                                num_agents=GridGame.numberofAgents,
                                index=i)
        ma.addAgent(agent)
    task=GridGameTask()

#     task=GridGameTask(gameType="GG1")
#     task=GridGameTask(gameType="GG2")
    task=GridGameTask(gameType="GG3")
    exp=EpisodicExperimentSG(task,ma)
    print "Rewards for agents at the end of episode:"
    for i in range(40000):
        rewards=exp.doEpisodes(number=1)
        print str(rewards[0][-1])
    