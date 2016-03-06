'''
Created on 2016/03/06

@author: takuya-hv2
'''
'''
Created on 2016/02/20

@author: takuya-hv2
'''

from pybrain.rl.environments import Environment
from pybrainSG.rl.environments.episodicSG import EpisodicTaskSG
import numpy as np
import copy

class GridGameTask(EpisodicTaskSG):
    '''' '''
    maximumTurn=30
    def __init__(self,gameType="GG1"):
        '''
        gameType: indicates game type an experiment perform: 
        [GG1:] simple coordinate game
        [GG2:] "Battle of the Sexes"
        [GG3:] "Chicken"
        See the following paper for detailed descriptions:
        https://www.aaai.org/Papers/ICML/2003/ICML03-034.pdf
        '''
        EpisodicTaskSG.__init__(self, GridGame(gameType))
    
    def reset(self):
        EpisodicTaskSG.reset(self)
        self.isGameFinished=False
        self.currentTurn=0
    
    def isFinished(self):
        if self.currentTurn > self.maximumTurn:
            return True
        return self.env.isReachGoal

    def getReward(self):
        self.currentTurn+=1
#         print "rew:"+str(self.env.getJointReward())
        return self.env.getJointReward()

class GridGame(Environment):
    availableActions=[0,1,2,3]#Corresponding to forward north, west, south, east respectively. 
    sizeofGlidWorld=3
    numberofGoals=2
    numberofAgents=2

    def __init__(self,gameType="GG1"):
        Environment.__init__(self)
        assert (gameType == "GG1") or (gameType == "GG2") or (gameType == "GG3"), "gameType should be either GG1, GG2, or GG3."
        self.gameType=gameType
        print "Game type: " + str(self.gameType)
    
    def getSensors(self):
        #State: locations of all agents
        state=[np.r_[self.agents[0],self.agents[1]],
               np.r_[self.agents[0],self.agents[1]]]
#         print "state:" + str(state)
        return state
    
    def performAction(self, action):
        tempPos=[]
        self.prevAgents=copy.deepcopy(self.agents)
#         print "act:" + str(action)
        self.isColide=False
        for i in range(GridGame.numberofAgents):
            tempPos.append(self.__move__(copy.deepcopy(self.agents[i]), action[i]))
        if not self.__isColideWithEachOther(tempPos):
            self.agents=tempPos
            
        
    def __move__(self,position, forward):
        if forward == 0:#Move North
            if self.gameType=="GG2":
                if (position[0] != 1 and position[1]==0) and (np.random.rand() < 0.5):
                    return position
            position[1]+=1
        elif forward == 1:#Move west
            position[0]-=1
        elif forward == 2:#Move south
            if self.gameType=="GG2":
                if (position[0] != 1 and position[1]==1) and (np.random.rand() < 0.5):
                    return position
            position[1]-=1
        elif forward == 3:#Move east
            position[0]+=1
        else:
            assert False, "Unexpected action"

        if position[0] >= GridGame.sizeofGlidWorld:
            position[0]=GridGame.sizeofGlidWorld-1
        if position[0] < 0:
            position[0]=0
        if position[1] >= GridGame.sizeofGlidWorld:
            position[1]=GridGame.sizeofGlidWorld-1
        if position[1] < 0:
            position[1]=0
        return position
    
    def __isColideWithEachOther(self,tempPos):
        if (tempPos[0][0] == tempPos[1][0]) and (tempPos[0][1] == tempPos[1][1]):
            if (tempPos[0][0] != self.goals[0][0]) or (tempPos[0][1] != self.goals[0][1]):
                self.isColide=True
                return True
            else:
                return False
        else:
            return False
        
    def __isReachGoal(self):
        #return boolean list, that determine if each agent reach each goal. 
        irGoal=[False,False]
        if (self.agents[0][0] == self.goals[0][0]) and (self.agents[0][1] == self.goals[0][1]):#For the first agent. 
            irGoal[0]=True
            self.isReachGoal=True
        if (self.agents[1][0] == self.goals[1][0]) and (self.agents[1][1] == self.goals[1][1]):#For the second agent. 
            irGoal[1]=True
            self.isReachGoal=True            
        return irGoal
    
    def reset(self):
        self.agents=[np.array([0,0]),
                     np.array([2,0])]
        self.prevAgents=[np.array([0,0]),
                     np.array([2,0])]

        self.isReachGoal=False
        if self.gameType == "GG1":
            self.goals=[np.array([2,2]),
                        np.array([0,2])]
        else:
            self.goals=[np.array([1,2]),
                        np.array([1,2])]
    
    def getJointReward(self):
        jointRew=[0,0]
        irGoal=self.__isReachGoal()
        if not (self.gameType == "GG3"):
            if irGoal[0]:
                jointRew[0]=100
            if irGoal[1]:
                jointRew[1]=100
        else:
            if  irGoal[0] and irGoal[1]:
                if self.prevAgents[0][0] == 1:
                    jointRew[0]= 125
                    jointRew[1]=100
                elif self.prevAgents[1][0] == 1:
                    jointRew[0]= 100
                    jointRew[1]=125
                else:
                    jointRew[0]= 120
                    jointRew[1]=120
            elif irGoal[0]:
                jointRew[0]=100
            elif irGoal[1]:
                jointRew[1]=100
        if self.isColide:
            jointRew[0]-=1
            jointRew[1]-=1
                
        return np.array(jointRew)
    



