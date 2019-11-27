# 
# ******************************** AI self driving car ****************************************
# 

# import libs

import numpy as np 
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of neural netwotk

class Netwotk(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Netwotk, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size , 30) # create a full connection bitween input layer and hidden layer
        self.fc2 = nn.Linear(30 , nb_action)  # create a full connection between hidden layer and the out put layer

    def forward(self , state):
        x = F.relu(self.fc1(state)) #activate the hidden neurons
        q_values = self.fc2(x)      # output neurons for the q-value
        return q_values

# Implimenting Experiance Replay

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        # if a list was {(1,2,3) , (4,5,6)} then zip(*list) => {(1,4) , (2,5) , (3,6)}
        samples = zip(*random.sample(self.memory , batch_size))
        return map(lambda x: Variable(torch.cat(x,0)),samples)

# Impliment deep Q learning

class Dqn():

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Netwotk(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0) #creating a tenser with a fake dimention / alrady have 5 dimentions => 3 signal + oriantation and -oriantation
        self.last_action =  0
        self.last_reword = 0



