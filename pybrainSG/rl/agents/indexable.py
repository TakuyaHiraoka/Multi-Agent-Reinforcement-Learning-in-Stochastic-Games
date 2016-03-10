'''
Created on 2016/02/19

@author: takuya-hv2
'''
__author__ = 'Takuya Hiraoka, takuya-h@is.naist.jp'

from pybrain.rl.agents.agent import Agent

class IndexableAgent(Agent):
    '''
    Agent which can be indexed.
    '''
    indexOfAgent=None
    
    def __init__(self, index=None):
        self.setIndexOfAgent(index)
        
    def setIndexOfAgent(self,index):
        """ set index to agent.
            :key index: index of agent
            :type index: integer
        """
        self.indexOfAgent=index
        
