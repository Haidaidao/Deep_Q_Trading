#Copyright (C) 2020 Salvatore Carta, Anselmo Ferreira, Alessandro Sebastian Podda, Diego Reforgiato Recupero, Antonio Sanna. All rights reserved.


#Imports the SPEnv library, which will perform the Agent actions themselves
from spEnv import SpEnv

#Callback used to print the results at each episode
from callback import ValidationCallback

#Keras library for the NN considered
from keras.models import Sequential

#Keras libraries for layers, activations and optimizers used
from keras.layers import Dense, Activation, Flatten
from keras.layers import LeakyReLU, PReLU
from keras.optimizers import Adam

#RL Agent 
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

#Mathematical operations used later
from math import floor

#Library to manipulate the dataset in a csv file
import pandas as pd

#Library used to manipulate time
import datetime

from trend import Trend
from macd import MACD
from lstmTrend import LSTMTrend
from AgentObject import AgentObject
import global_config

#Prefix of the name of the market (S&P500) files used to load the data
MK= global_config.MK
# MK="dax"

def getDate_Index(Frame, datesFrame, date):
    Frame['Date'] = pd.to_datetime(Frame['Date'], format='%m/%d/%Y')
    specific_date = datesFrame.loc[date,'Date']
    print("This date ",specific_date)
    specific_date = pd.to_datetime(specific_date, format='%m/%d/%Y')
    next_date = Frame[Frame['Date'] >= specific_date].iloc[0]['Date']
    print("Next date ",next_date)
    # print(str(datesFrame.loc[testMin[iteration],'Date']) + " ======= " + str(next_date))
    date_to_find = pd.to_datetime(next_date)
    index = Frame.index[Frame['Date'] == date_to_find].tolist()

    return index[0]

def getNumFile(agent,currentStartingPoint, walkSize, endingPoint, testSize, trainSize, validationSize):   
    iteration=-1
    while(currentStartingPoint+walkSize <= endingPoint):

        #Iteration is the current walk
        iteration+=1  
        trainMinLimit=None
        while(trainMinLimit is None):
            try:
                trainMinLimit = agent.sp.get_loc(currentStartingPoint)
                break
            except:
                currentStartingPoint+=datetime.timedelta(hours=1)

        trainMaxLimit=None
        while(trainMaxLimit is None):
            try:
                trainMaxLimit = agent.sp.get_loc(currentStartingPoint+trainSize)
            except:
                currentStartingPoint+=datetime.timedelta(hours=1)

        validMaxLimit=None
        while(validMaxLimit is None):
            try:
                validMaxLimit = agent.sp.get_loc(currentStartingPoint+trainSize+validationSize)
            except:
                currentStartingPoint+=datetime.timedelta(hours=1)

        testMaxLimit=None
        while(testMaxLimit is None):
            try:
                testMaxLimit = agent.sp.get_loc(currentStartingPoint+trainSize+validationSize+testSize)
            except:
                currentStartingPoint+=datetime.timedelta(hours=1)

        currentStartingPoint+=testSize
    return iteration

class DeepQTrading:

    def __init__(self, model, explorations, trainSize, validationSize, testSize, outputFile, begin, end, nbActions, isOnlyShort, ensembleFolderName, operationCost=0):

        self.isOnlyShort=isOnlyShort
        self.ensembleFolderName=ensembleFolderName

        self.agent=[]

        #Define the policy, explorations, actions and model as received by parameters
        self.policy = EpsGreedyQPolicy()
        self.explorations=explorations
        self.nbActions=nbActions
        self.model=model

        #Define the memory
        self.memory = SequentialMemory(limit=10000, window_length=1)

        #Instantiate the agent with parameters received
        self.agent.append(AgentObject(self.model, self.policy, self.nbActions, self.memory, 'Hour'))
        self.agent.append(AgentObject(self.model, self.policy, self.nbActions, self.memory, 'Day'))
        self.agent.append(AgentObject(self.model, self.policy, self.nbActions, self.memory, 'Week'))

        #Save the weights of the agents in the q.weights file
        #Save random weights
        for i in range(len(self.agent)):
            #Compile the agent with the adam optimizer and with the mean absolute error metric
            self.agent[i].agent.compile(Adam(lr=1e-3), metrics=['mae'])
            self.agent[i].agent.save_weights("q.weights", overwrite=True)

        #Define the current starting point as the initial date
        self.currentStartingPoint = begin

        #Define the training, validation and testing size as informed by the call
        #Train: 5 years
        #Validation: 6 months
        #Test: 6 months
        self.trainSize=trainSize
        self.validationSize=validationSize
        self.testSize=testSize

        #The walk size is simply summing up the train, validation and test sizes
        self.walkSize=trainSize+validationSize+testSize

        #Define the ending point as the final date (January 1st of 2010)
        self.endingPoint=end

        for i in range(len(self.agent)):
            self.agent[i].dates= pd.read_csv('./datasets/'+MK+self.agent[i].name+'.csv')
            self.agent[i].sp = pd.read_csv('./datasets/'+MK+self.agent[i].name+'.csv')
            #Convert the pandas format to date and time format
            self.agent[i].sp['Datetime'] = pd.to_datetime(self.agent[i].sp['Date'] + ' ' + self.agent[i].sp['Time'])
            #Set an index to Datetime on the pandas loaded dataset. Registers will be indexes through these values
            self.agent[i].sp = self.agent[i].sp.set_index('Datetime')
            #Drop Time and Date from the Dataset
            self.agent[i].sp = self.agent[i].sp.drop(['Time','Date'], axis=1)
            #Just the index considering date and time will be important, because date and time will be used to define the train,
            #validation and test for each walk
            self.agent[i].sp = self.agent[i].sp.index
            # print(self.agent[i].dates.loc[[0,2],'Date'])

        #Receives the operation cost, which is 0
        #Operation cost is the cost for long and short. It is defined as zero
        self.operationCost = operationCost

        #Call the callback for training, validation and test in order to show results for each episode
        self.trainer=ValidationCallback()
        self.train=ValidationCallback()
        self.validator=ValidationCallback()
        self.tester=ValidationCallback()
        self.outputFileName=outputFile

    

    def run(self):

        currentStartingPointTemp = self.currentStartingPoint
        numFile = getNumFile(self.agent[0], self.currentStartingPoint, self.walkSize, self.endingPoint, self.testSize, self.trainSize, self.validationSize)
        numFile = numFile + 1
        print(numFile)
        with open('numFile.txt', 'w', encoding='utf-8') as file:
            file.write(str(numFile-1))


        trainMin = [None]*(numFile+1)
        trainMax = [None]*(numFile+1)

        validMin = [None]*(numFile+1)
        validMax = [None]*(numFile+1)

        testMin = [None]*(numFile+1)
        testMax = [None]*(numFile+1)


        datesFrame = pd.read_csv('./datasets/'+MK+"Hour"+'.csv')
        weeksFrame = pd.read_csv('./datasets/'+MK+"Week"+'.csv')
        daysFrame = pd.read_csv('./datasets/'+MK+"Day"+'.csv')

        count = 0
        # print(getDate_Index(daysFrame, datesFrame, 2719))
        # print("===================")
        # print(getDate_Index(daysFrame, datesFrame, 17618))
        # return

        for i in range(len(self.agent)):
            #Initiates the environments,
            trainEnv=validEnv=testEnv=" "

            iteration=-1
            self.currentStartingPoint = currentStartingPointTemp
            name = self.agent[i].name
            index = i

            #While we did not pass through all the dates (i.e., while all the walks were not finished)
            #walk size is train+validation+test size
            #currentStarting point begins with begin date
            while(self.currentStartingPoint+self.walkSize <= self.endingPoint):

                #Iteration is the current walk
                iteration+=1
                if iteration < (numFile+1):
                    #Initiate the output file
                    self.outputFile=open(self.outputFileName+name+str(iteration+1)+".csv", "w+")
                    #write the first row of the csv
                    self.outputFile.write(
                        "Iteration,"+
                        "trainAccuracy,"+
                        "trainCoverage,"+
                        "trainReward,"+
                        "trainLong%,"+
                        "trainShort%,"+
                        "trainLongAcc,"+
                        "trainShortAcc,"+
                        "trainLongPrec,"+
                        "trainShortPrec,"+

                        "validationAccuracy,"+
                        "validationCoverage,"+
                        "validationReward,"+
                        "validationLong%,"+
                        "validationShort%,"+
                        "validationLongAcc,"+
                        "validationShortAcc,"+
                        "validLongPrec,"+
                        "validShortPrec,"+

                        "testAccuracy,"+
                        "testCoverage,"+
                        "testReward,"+
                        "testLong%,"+
                        "testShort%,"+
                        "testLongAcc,"+
                        "testShortAcc,"+
                        "testLongPrec,"+
                        "testShortPrec\n")


                    #Empty the memory and agent
                    del(self.agent[index].memory)
                    del(self.agent[index].agent)

                    #Define the memory and agent
                    #Memory is Sequential
                    self.agent[index].memory = SequentialMemory(limit=10000, window_length=1)
                    #Agent is initiated as passed through parameters
                    self.agent[index].agent = DQNAgent(model=self.model, policy=self.policy,  nb_actions=self.nbActions, memory=self.memory, nb_steps_warmup=200, target_model_update=1e-1,
                                            enable_double_dqn=True,enable_dueling_network=True)
                    #Compile the agent with Adam initialization
                    self.agent[index].agent.compile(Adam(lr=1e-3), metrics=['mae'])

                    #Load the weights saved before in a random way if it is the first time
                    self.agent[index].agent.load_weights("q.weights")
                    print("name: ", name)
                    print("iteration: ", iteration)
                    ########################################TRAINING STAGE########################################################

                    #The TrainMinLimit will be loaded as the initial date at the beginning, and will be updated later.
                    #If the initial date cannot be used, add 1 hour to the initial date and consider it the initial date
                    if name =="Hour":
                        trainMinLimit=None
                        while(trainMinLimit is None):
                            try:
                                trainMinLimit = self.agent[index].sp.get_loc(self.currentStartingPoint)
                                break
                            except:
                                self.currentStartingPoint+=datetime.timedelta(hours=1)
                        if name == "Hour":
                            trainMin[count] = trainMinLimit

                        #The TrainMaxLimit will be loaded as the interval between the initial date plus the training size.
                        #If the initial date cannot be used, add 1 hour to the initial date and consider it the initial date
                        trainMaxLimit=None
                        while(trainMaxLimit is None):
                            try:
                                trainMaxLimit = self.agent[index].sp.get_loc(self.currentStartingPoint+self.trainSize)
                            except:
                                self.currentStartingPoint+=datetime.timedelta(hours=1)
                        if name == "Hour":
                            trainMax[count] = trainMaxLimit

                        ########################################VALIDATION STAGE#######################################################
                        #The ValidMinLimit will be loaded as the next element of the TrainMax limit
                        validMinLimit=trainMaxLimit
                        if name == "Hour":
                            validMin[count] = validMinLimit

                        #The ValidMaxLimit will be loaded as the interval after the begin + train size +validation size
                        #If the initial date cannot be used, add 1 hour to the initial date and consider it the initial date
                        validMaxLimit=None
                        while(validMaxLimit is None):
                            try:
                                validMaxLimit = self.agent[index].sp.get_loc(self.currentStartingPoint+self.trainSize+self.validationSize)
                            except:
                                self.currentStartingPoint+=datetime.timedelta(hours=1)
                        if name == "Hour":
                            validMax[count] = validMaxLimit

                        ########################################TESTING STAGE########################################################
                        #The TestMinLimit will be loaded as the next element of ValidMaxlimit
                        testMinLimit=validMaxLimit
                        if name == "Hour":
                            testMin[count] = testMinLimit
                        #The testMaxLimit will be loaded as the interval after the begin + train size +validation size + Testsize
                        #If the initial date cannot be used, add 1 hour to the initial date and consider it the initial date
                        testMaxLimit=None
                        while(testMaxLimit is None):
                            try:
                                testMaxLimit = self.agent[index].sp.get_loc(self.currentStartingPoint+self.trainSize+self.validationSize+self.testSize)
                            except:
                                self.currentStartingPoint+=datetime.timedelta(hours=1)
                        if name == "Hour":
                            testMax[count] = testMaxLimit

                        if count<(numFile+1):
                            count = count +1
                    else:
                        
                        if name == "Week":
                            trainMinLimit = getDate_Index(weeksFrame, datesFrame, trainMin[iteration])
                            trainMaxLimit = getDate_Index(weeksFrame, datesFrame, trainMax[iteration])

                            validMinLimit = getDate_Index(weeksFrame, datesFrame, validMin[iteration])
                            validMaxLimit = getDate_Index(weeksFrame, datesFrame, validMax[iteration])
                        
                            testMinLimit = getDate_Index(weeksFrame, datesFrame, testMin[iteration])
                            testMaxLimit = getDate_Index(weeksFrame, datesFrame, testMax[iteration])
                        else:
                            trainMinLimit = getDate_Index(daysFrame, datesFrame, trainMin[iteration])
                            trainMaxLimit = getDate_Index(daysFrame, datesFrame, trainMax[iteration])

                            validMinLimit = getDate_Index(daysFrame, datesFrame, validMin[iteration])
                            validMaxLimit = getDate_Index(daysFrame, datesFrame, validMax[iteration])
                        
                            testMinLimit = getDate_Index(daysFrame, datesFrame, testMin[iteration])
                            testMaxLimit = getDate_Index(daysFrame, datesFrame, testMax[iteration])                   

                    #Separate the Validation and testing data according to the limits found before
                    #Prepare the training and validation files for saving them later
                    ensambleTrain=pd.DataFrame(index=self.agent[index].dates[trainMinLimit:trainMaxLimit].loc[:,'Date'].drop_duplicates().tolist())
                    ensambleValid=pd.DataFrame(index=self.agent[index].dates[validMinLimit:validMaxLimit].loc[:,'Date'].drop_duplicates().tolist())
                    ensambleTest=pd.DataFrame(index=self.agent[index].dates[testMinLimit:testMaxLimit].loc[:,'Date'].drop_duplicates().tolist())

                    #Put the name of the index for validation and testing
                    ensambleTrain.index.name='Date'
                    ensambleValid.index.name='Date'
                    ensambleTest.index.name='Date'

                    if name == "Hour":
                        #Explorations are epochs considered, or how many times the agent will play the game.
                        for eps in self.explorations:

                            #policy will be 0.2, so the randomness of predictions (actions) will happen with 20% of probability
                            self.policy.eps = eps[0]

                            #there will be 100 iterations (epochs), or eps[1])

                            for i in range(0,eps[1]):
                                del(trainEnv)

                                #Define the training, validation and testing environments with their respective callbacks
                                print("train")
                                print("minLimit=trainMinLimit,maxLimit=trainMaxLimit : " +  str(trainMinLimit) + " == " +  str(trainMaxLimit))
                                trainEnv = SpEnv(operationCost=self.operationCost,minLimit=trainMinLimit,maxLimit=trainMaxLimit,callback=self.trainer,isOnlyShort=self.isOnlyShort, ensamble=ensambleTrain,columnName="iteration"+str(i), name=name)
                                del(validEnv)
                                print("valid")
                                print("minLimit=validMinLimit,maxLimit=validMaxLimit : " +  str(validMinLimit) + " == " +  str(validMaxLimit))
                                validEnv=SpEnv(operationCost=self.operationCost, minLimit=validMinLimit,maxLimit=validMaxLimit,callback=self.validator,isOnlyShort=self.isOnlyShort,ensamble=ensambleValid,columnName="iteration"+str(i), name=name)
                                del(testEnv)
                                print("test")
                                print("minLimit=testMinLimit,maxLimit=testMaxLimit : " +  str(testMinLimit) + " == " +  str(testMaxLimit))
                                testEnv=SpEnv(operationCost=self.operationCost,minLimit=testMinLimit,maxLimit=testMaxLimit,callback=self.tester,isOnlyShort=self.isOnlyShort,ensamble=ensambleTest,columnName="iteration"+str(i), name=name)

                                #Reset the callback
                                self.trainer.reset()
                                self.validator.reset()
                                self.tester.reset()

                                #Reset the training environment
                                trainEnv.resetEnv()
                                #Train the agent
                                self.agent[index].agent.fit(trainEnv,nb_steps=floor(self.trainSize.days-self.trainSize.days*0.2),visualize=False,verbose=0)
                                #Get the info from the train callback
                                (_,trainCoverage,trainAccuracy,trainReward,trainLongPerc,trainShortPerc,trainLongAcc,trainShortAcc,trainLongPrec,trainShortPrec)=self.trainer.getInfo()
                                #Print Callback values on the screen
                                print(str(i) + " TRAIN:  acc: " + str(trainAccuracy)+ " cov: " + str(trainCoverage)+ " rew: " + str(trainReward))

                                #Reset the validation environment
                                validEnv.resetEnv()
                                #Test the agent on validation data
                                self.agent[index].agent.test(validEnv,nb_episodes=floor(self.validationSize.days-self.validationSize.days*0.2),visualize=False,verbose=0)
                                #Get the info from the validation callback
                                (_,validCoverage,validAccuracy,validReward,validLongPerc,validShortPerc,validLongAcc,validShortAcc,validLongPrec,validShortPrec)=self.validator.getInfo()
                                #Print callback values on the screen
                                print(str(i) + " VALID:  acc: " + str(validAccuracy)+ " cov: " + str(validCoverage)+ " rew: " + str(validReward))

                                #Reset the testing environment
                                testEnv.resetEnv()
                                #Test the agent on testing data
                                self.agent[index].agent.test(testEnv,nb_episodes=floor(self.validationSize.days-self.validationSize.days*0.2),visualize=False,verbose=0)
                                #Get the info from the testing callback
                                (_,testCoverage,testAccuracy,testReward,testLongPerc,testShortPerc,testLongAcc,testShortAcc,testLongPrec,testShortPrec)=self.tester.getInfo()
                                #Print callback values on the screen
                                print(str(i) + " TEST:  acc: " + str(testAccuracy)+ " cov: " + str(testCoverage)+ " rew: " + str(testReward))
                                print(" ")

                                #write the walk data on the text file
                                self.outputFile.write(
                                    str(i)+","+
                                    str(trainAccuracy)+","+
                                    str(trainCoverage)+","+
                                    str(trainReward)+","+
                                    str(trainLongPerc)+","+
                                    str(trainShortPerc)+","+
                                    str(trainLongAcc)+","+
                                    str(trainShortAcc)+","+
                                    str(trainLongPrec)+","+
                                    str(trainShortPrec)+","+

                                    str(validAccuracy)+","+
                                    str(validCoverage)+","+
                                    str(validReward)+","+
                                    str(validLongPerc)+","+
                                    str(validShortPerc)+","+
                                    str(validLongAcc)+","+
                                    str(validShortAcc)+","+
                                    str(validLongPrec)+","+
                                    str(validShortPrec)+","+

                                    str(testAccuracy)+","+
                                    str(testCoverage)+","+
                                    str(testReward)+","+
                                    str(testLongPerc)+","+
                                    str(testShortPerc)+","+
                                    str(testLongAcc)+","+
                                    str(testShortAcc)+","+
                                    str(testLongPrec)+","+
                                    str(testShortPrec)+"\n")

                        #Close the file
                        self.outputFile.close()

                        #Write validation and Testing data into files
                        #Save the files for processing later with the ensemble considering the 100 epochs
                        ensambleTrain.to_csv("./Output/ensemble/"+self.ensembleFolderName+"/walk"+self.agent[index].name+str(iteration)+"ensemble_train.csv")
                        ensambleValid.to_csv("./Output/ensemble/"+self.ensembleFolderName+"/walk"+self.agent[index].name+str(iteration)+"ensemble_valid.csv")
                        ensambleTest.to_csv("./Output/ensemble/"+self.ensembleFolderName+"/walk"+self.agent[index].name+str(iteration)+"ensemble_test.csv")
                    else:
                        train = LSTMTrend(iteration = iteration, minLimit=trainMinLimit,maxLimit=trainMaxLimit, name = name ,type = "train")
                        train.writeFile()
                        valid = LSTMTrend(iteration = iteration, minLimit=validMinLimit,maxLimit=validMaxLimit, name = name ,type = "valid")
                        valid.writeFile()
                        test  = LSTMTrend(iteration = iteration, minLimit=testMinLimit,maxLimit=testMaxLimit, name = name ,type = "test")
                        test.writeFile()
                       
                    #For the next walk, the current starting point will be the current starting point + the test size
                    #It means that, for the next walk, the training data will start 6 months after the training data of
                    #the previous walk
                self.currentStartingPoint+=self.testSize



    #Function to end the Agent
    def end(self):
        print("END")