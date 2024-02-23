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


class AgentObject:
    def __init__(self, model, policy, nbActions, memory, name):
        self.model = model
        self.policy = policy
        self.nbActions = nbActions
        self.memory = memory
        self.agent = DQNAgent(model=self.model, policy=self.policy,  nb_actions=self.nbActions, memory=self.memory, nb_steps_warmup=200, target_model_update=1e-1,
                                    enable_double_dqn=True,enable_dueling_network=True)
        self.dates = pd.DataFrame()
        self.sp = pd.DataFrame()
        self.name = name


