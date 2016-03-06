'''
Created on 2016/02/20

@author: takuya-hv2
'''

from pybrain.rl.environments import Environment
from pybrainSG.rl.environments.episodicSG import EpisodicTaskSG
import numpy as np

#Integrated to gridgames.py in future.
class HuntingGameTask(EpisodicTaskSG):
    ''''Agents hunt animals in grid world. If all agents gather into particular place where more than one animal stay, 
    hunting is succeeded and agents are rewarded. 
    Agents are punished as turn passes. 
    '''
    isGameFinished=False
    maximumTurn=10
    currentTurn=0
    def __init__(self,task=None):
        if task == None:
            task=HuntingGame()
        EpisodicTaskSG.__init__(self, task)
    
    def reset(self):
        EpisodicTaskSG.reset(self)
        self.isGameFinished=False
        self.currentTurn=0
    
    def isFinished(self):
        return self.isGameFinished

    def getReward(self):
        jointReward=self.env.getJointReward()
        if (self.env.getJointReward()[0] > 0):
            self.isGameFinished=True
            
        #Time pressure
        jointReward=jointReward-1
        if (self.currentTurn >= HuntingGameTask.maximumTurn):
            self.isGameFinished=True
        #print str(jointReward[0])#+", " + str(jointReward[1])
        self.currentTurn+=1
        return jointReward 


class HuntingGame(Environment):
    availableActions=[0,1,2,3,4]#Corresponding to forward north, west, south, east, stay respectively. 
    sizeofGlidWorld=3
    numberofAnimals=1
    numberofAgents=2
    animals=None
    agents=None
    
    def getSensors(self):
        for i in range(HuntingGame.numberofAnimals):
            if np.random.rand() > 0.8:
                self.animals[i]=self.__move__(self.animals[i], np.random.randint(5))
        stateTemp1=np.append(self.animals.flatten(),self.agents.flatten())
        stateTemp2=np.append(stateTemp1,np.ones(1))
        stateTemp3=[]
        for _ in range(self.numberofAgents):
            stateTemp3.append(stateTemp2)
        
        return stateTemp3
    
    def performAction(self, action):
        for i in range(HuntingGame.numberofAgents):
            self.agents[i]=self.__move__(self.agents[i], action[i])
            
    def isSucceedHunting(self):
        #return true only if all agent gather in one place where animal exists
        for i in range(HuntingGame.numberofAgents):
            for j in range(HuntingGame.numberofAgents):
                if(self.agents[i][0] != self.agents[j][0]) or (self.agents[i][1] != self.agents[j][1]):
                    return False
        for k in range(HuntingGame.numberofAnimals):
            if(self.agents[0][0] == self.animals[k][0]) and (self.agents[0][1] == self.animals[k][1]):
                return True
        return False
        
    def __move__(self,position, forward):
        if forward == 0:#Move North
            position[1]+=1
        elif forward == 1:#Move west
            position[0]-=1
        elif forward == 2:#Move south
            position[1]-=1
        elif forward == 3:#Move east
            position[0]+=1
        elif forward == 4:#stay here
            return position
        else:
            assert False, "Unexpected action"

        if position[0] >= HuntingGame.sizeofGlidWorld:
            position[0]=HuntingGame.sizeofGlidWorld-1
        if position[0] < 0:
            position[0]=0
        if position[1] >= HuntingGame.sizeofGlidWorld:
            position[1]=HuntingGame.sizeofGlidWorld-1
        if position[1] < 0:
            position[1]=0
        return position
                
    def reset(self):
#         self.animals=np.random.randint(HuntingGame.sizeofGlidWorld,size=(HuntingGame.numberofAnimals,2))
        self.animals=np.zeros((HuntingGame.numberofAnimals,2))
        self.agents=np.random.randint(HuntingGame.sizeofGlidWorld,size=(HuntingGame.numberofAgents,2))
#         self.agents=np.ones((HuntingGame.numberofAgents,2))

    
    def getJointReward(self):
        if self.isSucceedHunting():
            return np.ones(self.numberofAgents)*10
        return np.zeros(self.numberofAgents)
    



