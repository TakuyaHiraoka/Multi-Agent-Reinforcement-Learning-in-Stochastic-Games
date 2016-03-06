'''
Created on 2016/02/26

@author: takuya-hv2
'''
from pybrainSG.rl.leaners.valuebased.indexablevaluebased import IndexableValueBasedLearner
from pybrain.utilities import r_argmax
import numpy as np
from pybrain.utilities import abstractMethod
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.rprop import RPropMinusTrainer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import one_to_n
from pybrain.structure.modules import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork
import copy
from scipy import zeros, exp, clip
from scipy.optimize._linprog import linprog
import warnings
from multiprocessing import Process, Queue


class CEQ_FA(IndexableValueBasedLearner):
    """ 
    Correlated Q (with function approximation):
    http://www.aaai.org/Papers/Symposia/Spring/2002/SS-02-02/SS02-02-012.pdf
    """
    
    learningRate = 0.2      # aka alpha: make sure this is being decreased by calls from the learning agent!
    learningRateDecay = 100 # aka n_0, but counting decay-calls
    
    randomInit = True
    
    rewardDiscount = 0.99 # aka gamma
    
    batchMode = False
    passNextAction = False # for the _updateWeights method
    
    def __init__(self, num_features, num_actions, num_agents, indexOfAgent):
        IndexableValueBasedLearner.__init__(self, indexOfAgent)
        self.explorer = None        
        self.num_actions = num_actions
        self.num_features = num_features
        self.num_agents=num_agents
        self.reset()
        self.ownerAgentProperties["requireOtherAgentsState"]=False
        self.ownerAgentProperties["requireJointAction"]=True
        self.ownerAgentProperties["requireJointReward"]=True
        assert self.num_agents == np.size(self.num_actions, axis=0), "Size of 1st row of action array should be equal to number of agent. "
        
    def _qValues(self, state):
        """ Return vector of probability of policy for all actions, 
        given the state(-features). """
        abstractMethod()
           
    def _greedyAction(self, state):
        return r_argmax(self._qValues(state))
    
    def _greedyPolicy(self, state):
        tmp = zeros(self.num_actions[self.indexOfAgent])
        tmp[self._greedyAction(state)] = 1
        return tmp
    
    def _boltzmannPolicy(self, state, temperature=1.):
        tmp = self._qValues(state)
        return CEQ_FA._boltzmannProbs(tmp, temperature)
    
    @staticmethod
    def _boltzmannProbs(qvalues, temperature=1.):
        if temperature == 0:
            tmp = zeros(len(qvalues))        
            tmp[r_argmax(qvalues)] = 1.
        else:
            tmp = qvalues / temperature            
            tmp -= max(tmp)
            tmp = exp(clip(tmp, -20, 0))
        return tmp / sum(tmp)

    def reset(self):
        IndexableValueBasedLearner.reset(self)        
        self._callcount = 0
        self.newEpisode()
    
    def newEpisode(self):  
        IndexableValueBasedLearner.newEpisode(self)      
        self._callcount += 1
        self.learningRate *= ((self.learningRateDecay + self._callcount) 
                              / (self.learningRateDecay + self._callcount + 1.))
    
    def _updateWeights(self, state, action, reward, next_state):
        '''
        Expected to update Q-value approximator. 
        '''
        abstractMethod()
        
        
class CEQ_Lin(CEQ_FA):
    '''
    CEQ with linear function approximation. 
    '''
    def __init__(self, num_features, num_actions, num_agents, indexOfAgent=None):    
        CEQ_FA.__init__(self, num_features, num_actions, num_agents, indexOfAgent)
        self.possibleJointAction, self.w4ActIndexing = self._initJointActAndItsIndex(num_agents, num_actions)
        self.numJointAct=np.size(self.possibleJointAction, axis=0)
        self.linQ=[]
        self.actionDiminInput=0
        for i in range(self.num_agents):
            self.actionDiminInput+=self.num_actions[i]
        for i in range(self.num_agents):
            self.linQ.append(buildNetwork(num_features + self.actionDiminInput, 1, outclass = LinearLayer))
        self.actionVecDic={}
    
    def _initJointActAndItsIndex(self, num_agents, num_actions):
        numJointAct=1
        w4ActIndexing=np.zeros(num_agents)
        for index in range(len(num_actions)):
            numJointAct*=num_actions[index]
        temp=numJointAct
        for index in range(np.size(num_actions,axis=0)):
            temp/=num_actions[index]
            w4ActIndexing[index]=(temp)
        possibleJointAction=[[]]
        for i in range(num_agents):
            temp=[]
            for j in range(num_actions[i]):
                for k in range(len(possibleJointAction)):
                    temp2=copy.deepcopy(possibleJointAction[k])
                    temp2.append(j)
                    temp.append(temp2)
            possibleJointAction=temp
        possibleJointAction.sort()
        possibleJointAction=np.array(possibleJointAction)
        return possibleJointAction, w4ActIndexing
    
    def _qValues(self, state):
        """ Return vector of q-values for all actions, 
        given the state(-features). """
        qValues=self._qValuesForAllPossibleJointAction(state)
        eq=findCorrelatedEquilibrium(self.num_agents, self.num_actions, qValues, self.possibleJointAction, self.w4ActIndexing)
        return np.array(self._qValuesForEachActionOfAgent(state, eq, self.indexOfAgent)).reshape(self.num_actions[self.indexOfAgent])
    
    def _updateWeights(self, state, action, reward, next_state):
        """ state and next_state are vectors, action is an integer. """
        #update Q-value function approximator
        qValuesNext=self._qValuesForAllPossibleJointAction(next_state)
        eqNext=findCorrelatedEquilibrium(self.num_agents, self.num_actions, qValuesNext, self.possibleJointAction,self.w4ActIndexing)
        #Learn
        inp=self._EncodeStateAndJointActionIntoInputVector(state, action)
        for i in range(self.num_agents):
            target=reward[i] + self.rewardDiscount * max(self._qValuesForEachActionOfAgent(next_state, eqNext, i))
            self.trainer4LinQ=BackpropTrainer(self.linQ[i],learningrate=self.learningRate,weightdecay=0.0)
            ds = SupervisedDataSet(self.num_features+self.actionDiminInput,1)
            ds.addSample(inp, target)
            self.trainer4LinQ.trainOnDataset(ds)
    
    def _qValuesForAllPossibleJointAction(self, state):
        qValues=[]
        for iAgent in range(self.num_agents):
            qValuesIthAgent=[]
            for jointAct in self.possibleJointAction:
                val=np.array(self.linQ[iAgent].activate(self._EncodeStateAndJointActionIntoInputVector(state, jointAct)))
                qValuesIthAgent.append(val)
            qValues.append(qValuesIthAgent)
        return qValues#QValues for all possible joint actions for each agents [numAgents][index of joint act in list]
    
    def _qValuesForEachActionOfAgent(self, state, CEq, iAgent):
        qValuesForeachAct=[]
        for iAct in range(self.num_actions[iAgent]):
            expQ=0.0
            sumP=0.0
            numPJA=0.0
            for jointAct in self.possibleJointAction:
                if iAct == int(jointAct[iAgent]):
                    sumP+=CEq[int(np.dot(self.w4ActIndexing, jointAct))]
                    numPJA+=1.0
            for jointAct in self.possibleJointAction:
                if iAct == int(jointAct[iAgent]):
                    if sumP > 0.00001:
                        prob=CEq[int(np.dot(self.w4ActIndexing, jointAct))]
                        if prob > 0.0:
                            Q=self.linQ[iAgent].activate(self._EncodeStateAndJointActionIntoInputVector(state, jointAct))
                            expQ+=(prob/sumP)*Q[0]
                    else:
                        Q=self.linQ[iAgent].activate(self._EncodeStateAndJointActionIntoInputVector(state, jointAct))
                        expQ+=(1.0/numPJA)*Q[0]
            qValuesForeachAct.append(expQ)
        return qValuesForeachAct
    
    
    def _EncodeStateAndJointActionIntoInputVector(self, state, jointAct):
        index=int(np.dot(self.w4ActIndexing, jointAct))
        if index in self.actionVecDic:
            return np.r_[state, self.actionVecDic[index]]
        else:
            iVector=np.array([])
            for iAgent in range(len(jointAct)):
                iVector=np.r_[iVector, one_to_n(jointAct[iAgent], self.num_actions[iAgent])]
            self.actionVecDic[index]=iVector
            return np.r_[state, self.actionVecDic[index]]
        
        
        
class NFCEQ(CEQ_Lin):
    '''Neural fitted Q iteration version. '''
    def __init__(self, num_features, num_actions, num_agents, max_epochs=20, indexOfAgent=None, validateMultiProc=True):
        CEQ_Lin.__init__(self, num_features, num_actions, num_agents, indexOfAgent)
        self.max_epochs=max_epochs
        self.linQ=[]#update
        for _ in range(self.num_agents):
            self.linQ.append(buildNetwork(num_features + self.actionDiminInput, (num_features + self.actionDiminInput), 1, hiddenclass=SigmoidLayer, outclass = LinearLayer))
        self.isFirstLerning=True
        self.validateMultiProc=validateMultiProc

    def _updateWeights(self, state, action, reward, next_state):
        """ state and next_state are vectors, action is an integer. """
        pass
    def learn(self):
        # convert reinforcement dataset to NFQ supervised dataset
        supervised = []
        dats=[]#[seq index][turn]=[state,jointAct,jointReward]
        for i in range(self.num_agents):
            supervised.append(SupervisedDataSet(self.num_features+self.actionDiminInput, 1))
        for i in range(self.dataset[self.indexOfAgent].getNumSequences()):            
            seq=[]
            for j in range(len(self.dataset[self.indexOfAgent].getSequence(i)[0])):
                state=self.dataset[self.indexOfAgent].getSequence(i)[0][j]
                jointAct=[]
                jointReward=[]
                for k in range(self.num_agents):
                    jointAct.append(self.dataset[k].getSequence(i)[1][j][0])
                    jointReward.append(self.dataset[k].getSequence(i)[2][j][0])
                seq.append([state, jointAct, jointReward])
            dats.append(seq)
        #prepare data set
        for i in range(self.num_agents):
            for seq in dats:
                lastexperience = None
                for sarPair in seq:
                    state = sarPair[0]
                    action = sarPair[1]
                    reward = sarPair[2]
                    if not lastexperience:
                        # delay each experience in sequence by one
                        lastexperience = (state, action, reward)
                        continue
                    # use experience from last timestep to do Q update
                    (state_, action_, reward_) = lastexperience
                    
                    #update Q-value function approximator
                    qValuesNext=self._qValuesForAllPossibleJointAction(state)
                    eqNext=findCorrelatedEquilibrium(self.num_agents, self.num_actions, qValuesNext, self.possibleJointAction,self.w4ActIndexing)
                    #Learn
                    inp=self._EncodeStateAndJointActionIntoInputVector(state_, action_)
                    if self.isFirstLerning:
                        target=reward_[i]
                    else:
                        target=reward_[i] + self.rewardDiscount * max(self._qValuesForEachActionOfAgent(state, eqNext, i))
                    target=np.array([target])
                    supervised[i].addSample(inp, target)
                    # update last experience with current one
                    lastexperience = (state, action, reward)
        if self.isFirstLerning:
            self.isFirstLerning=False
            
        procTrainers=[]
        qResult=Queue()
        for i in range(self.num_agents):
            trainer=RPropMinusTrainer(self.linQ[i],dataset=supervised[i], 
                                      batchlearning=True, 
                                      verbose=False, 
                                      )
            if not self.validateMultiProc:
                trainer.trainUntilConvergence(maxEpochs=self.max_epochs,verbose=False)
            else:
                procTrainers.append(Process(target=self._learningQfunction, kwargs={"trainer":trainer,"i":i,"q":qResult}))
        if self.validateMultiProc:
            for proc in procTrainers:
                proc.start()
            for i in range(self.num_agents):
                res=qResult.get()
                self.linQ[res[0]]=res[1]
    
    def _learningQfunction(self, trainer,i,q):
        #Re-builde networks is required in multiprocessing environments. 
        params=trainer.module.params
        trainer.module=buildNetwork(self.num_features + self.actionDiminInput, (self.num_features + self.actionDiminInput), 1, hiddenclass=SigmoidLayer, outclass = LinearLayer)
        trainer.module._setParameters(params)
        trainer.trainUntilConvergence(maxEpochs=self.max_epochs,verbose=False)
        q.put([i,trainer.module])






def findCorrelatedEquilibrium(numAgent, numAction, Qvalues, possibleJointAction, w4ActIndexing):
    '''
    Given a list of all possible joint action, and its QValue table, 
    this function find correlated equilibrium based on the linear programming. 
    #In current implementation, the objective function, to determine the identical equilibrium, is "republican" function. 
    '''
    numJointAct=np.size(possibleJointAction,axis=0)
    STs=[]#constraints for LP
    for iAgent in range(numAgent): 
#         print "==================== Agent " + str(iAgent) + "==============="
        vecQ=Qvalues[iAgent]
        eCumdeltaOutCome=np.zeros(numJointAct)
        for ithAgentsOptAct in range(numAction[iAgent]):
            #Calculate expected Q-Value when agent follow its optimal action "ithAgentsOptAct". 
            eOutcomeInOpt=np.zeros(numJointAct)
            for jointAction in possibleJointAction:
                if ithAgentsOptAct == jointAction[iAgent]:
                    index=int(np.dot(w4ActIndexing, jointAction))
                    eOutcomeInOpt[index]=vecQ[index]
            #Calculate expected Q-Value when agent follow its non-optimal action "ithAgentsOptAct". 
            for ithAgentsNonOptAct in range(numAction[iAgent]):
                if ithAgentsNonOptAct == ithAgentsOptAct:
                    continue
                eOutcomeInNonOpt=np.zeros(numJointAct)
                for jointAction in possibleJointAction:
                    if (ithAgentsOptAct != jointAction[iAgent]) and (ithAgentsNonOptAct == jointAction[iAgent]):
                        jointActionWithNonOptimal=copy.deepcopy(jointAction)
                        jointActionWithNonOptimal[iAgent]=ithAgentsOptAct
                        index1=int(np.dot(w4ActIndexing, jointActionWithNonOptimal))
                        index2=int(np.dot(w4ActIndexing, jointAction))
                        eOutcomeInNonOpt[index1]=vecQ[index2]
                eCumdeltaOutCome = eCumdeltaOutCome + (eOutcomeInOpt - eOutcomeInNonOpt)
        STs.append(eCumdeltaOutCome)
            
    #All all possible ith agent action
    for i in range(numJointAct):
        t=np.zeros(numJointAct)
        t[i]=1.0
        STs.append(t)
    STs=np.array(STs)*(-1)
    #Constraints (uneq.)
    b_ub=np.zeros(np.size(STs,axis=0))
    #Constraints (eq.)    
    A_eq=np.ones((1,numJointAct))
    b_eq=np.ones(1)
    #Objective function    
    c=np.zeros(numJointAct)
    for iAgent in range(numAgent):
        #find maximum Q value in each decision making. 
        vecQ=np.array(Qvalues[iAgent]).reshape(numJointAct)
        for jointAction in possibleJointAction:
            index=int(np.dot(w4ActIndexing, jointAction))
            if c[index] < vecQ[index]:
                c[index] = vecQ[index]
    c*=-1
    #Implement linear programing with scipy library
    res=linprog(c=c, A_ub=STs, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=None, method='simplex', callback=None, options=None)
    if not res.success:
        warnings.warn("LP was failed uniform probability was set .")
        res.x = np.ones(numJointAct)/(numJointAct)
    return res.x
