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

