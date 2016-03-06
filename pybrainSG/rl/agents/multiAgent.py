'''
Created on 2016/02/19

@author: takuya-hv2
'''

from pybrain.rl.agents.agent import Agent
from pybrainSG.rl.agents.indexable import IndexableAgent
from pybrainSG.rl.agents.loggingSG import LoggingAgentSG
import numpy as np
from multiprocessing import Process, Queue
import copy_reg
import types

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

class MultiAgent(Agent):
    '''
    This class defines set of agents. 
    Each agent should be instance of IndexableAgent or its subclass.
    '''
    agentSet=[]
    
    def __init__(self):
        Agent.__init__(self)
        self.agentSet=[]
    
    def integrateObservation(self, obs):
        """ Integrate the current observation of the environment.
            :arg obs: The last observation returned from the environment
            :type obs: by default, this is assumed to be a numpy array of doubles
        """
        for index in range(len(self.agentSet)):
            if self.agentSet[index].getProperty()["requireOtherAgentsState"]:
                self.agentSet[index].integrateObservation(obs)
            else:
                self.agentSet[index].integrateObservation(obs[index])
        
    def getJointAction(self):
        """ Return a chosen joint-action.
            :rtype: by default, this is assumed to ba a numpy array of integers that correspond to particular action at each.
        """
        jointAction=np.zeros(len(self.agentSet), dtype=np.int)
        for index in range(len(self.agentSet)):
            jointAction[index]=self.agentSet[index].getAction()
        for index in range(len(self.agentSet)):
            if isinstance(self.agentSet[index], LoggingAgentSG) and self.agentSet[index].getProperty()["requireJointAction"]:
                self.agentSet[index].lastaction=jointAction
            else:
                self.agentSet[index].lastaction=jointAction[index]        
        return jointAction    
    
    def _getAction(self,q, agent, index):
        act=agent.getAction()
        q.put([index,act])
    
    def giveJointReward(self, r):
        """ give joint-teward to all agents.
            :key r: joint reward
            :type r: numpy array of doubles
        """
        for index in range(len(self.agentSet)):
            if self.agentSet[index].getProperty()["requireJointReward"]:
                self.agentSet[index].giveReward(r)
            else:
                self.agentSet[index].giveReward(r[index])                

    def reset(self):
        for agent in self.agentSet:
            agent.reset()

    def learn(self, episodes=1):
        procs=[]
        i=0
        qResult=Queue()
        for agent in self.agentSet:
            procs.append(Process(target=self._paraLearn, kwargs={"agent":agent,"episodes":episodes,"qResult":qResult}))
            i+=1
        for proc in procs:
            proc.start()
        for _ in range(len(self.agentSet)):
            res=qResult.get()
            self.agentSet[res[0]]=res[1]

    def _paraLearn(self, agent, episodes, qResult):
        agent.learn(episodes)
        qResult.put([agent.indexOfAgent, agent])
        
    def newEpisode(self):
        for agent in self.agentSet:
            agent.newEpisode()
    
    def addAgent(self, agent):
        assert isinstance(agent, IndexableAgent), "agent should be IndxableAgent class or its subclass."
        assert agent.indexOfAgent is not None, "Index should be identified"
        if len(self.agentSet) ==0:
            assert agent.indexOfAgent==0, "Illegal indexing."
        else:
            ind=0
            for elem in self.agentSet:
                assert ind == (elem.indexOfAgent), "Illegal indexing."
                ind+=1
            assert agent.indexOfAgent==ind, "Illegal indexing."
        self.agentSet.append(agent)
    
    def popAgent(self, index):
        agent=self.agentSet.pop(index)
        agent.setIndexOfAgent(None)
        
    