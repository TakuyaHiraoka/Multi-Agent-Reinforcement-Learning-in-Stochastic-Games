from scipy import r_
from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.rprop import RPropMinusTrainer
from pybrain.utilities import one_to_n
from pybrainSG.rl.leaners.valuebased.learnerfaSG import IndexableValueBasedLearner
from pybrain.tools.shortcuts import buildNetwork

class NFQ_SG(IndexableValueBasedLearner):#Mod. version
    """ 
    Stochastic game version of Neural-fitted Q-iteration
    """

    def __init__(self, maxEpochs=20, indexOfAgent=None,):
        ValueBasedLearner.__init__(self)
        self.gamma = 0.9
        self.maxEpochs = maxEpochs
        #
        self.ownerAgentProperties["requireOtherAgentsState"]=False
        self.ownerAgentProperties["requireJointAction"]=False
        self.ownerAgentProperties["requireJointReward"]=False
        self.isFirstLerning=True
        
    def learn(self):
        # convert reinforcement dataset to NFQ supervised dataset
        supervised = SupervisedDataSet(self.module.network.indim, 1)
        for seq in self.dataset[self.indexOfAgent]:
            lastexperience = None
            for state, action, reward in seq:
                if not lastexperience:
                    # delay each experience in sequence by one
                    lastexperience = (state, action, reward)
                    continue
                
                # use experience from last timestep to do Q update
                (state_, action_, reward_) = lastexperience
                
                Q = self.module.getValue(state_, action_[0])
                
                inp = r_[state_, one_to_n(action_[0], self.module.numActions)]
                if self.isFirstLerning:
                    tgt = reward_
                else:
                    tgt = Q + 0.5*(reward_ + self.gamma * max(self.module.getActionValues(state)) - Q)
                supervised.addSample(inp, tgt)
                
                #for reward normalization
                
                # update last experience with current one
                lastexperience = (state, action, reward)
                
        #Re-building netowrks is required in multiprocessing environments. 
        params=self.module.network.params
        self.module.network=buildNetwork(self.module.indim+self.module.numActions, 
                                         self.module.indim+self.module.numActions, 
                                         1)
        self.module.network._setParameters(params)
        
        # train module with backprop/rprop on dataset
        trainer = RPropMinusTrainer(self.module.network, dataset=supervised, batchlearning=True, verbose=False)#, weightdecay=0.01)
        trainer.trainUntilConvergence(maxEpochs=self.maxEpochs)
        if self.isFirstLerning:
            self.isFirstLerning=False
        # alternative: backprop, was not as stable as rprop
        # trainer = BackpropTrainer(self.module.network, dataset=supervised, learningrate=0.005, batchlearning=True, verbose=True)
        # trainer.trainUntilConvergence(maxEpochs=self.maxEpochs)



