from copy import deepcopy
from math import log
import os
import pygame
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import time
import numpy as np
from cnn import *
from constants import *
from game import GameWrapper
import random
import matplotlib
from state import GameState
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, resize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch

import torch

class DQNCNN(nn.Module):
    def __init__(self):
        super(DQNCNN, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(64 * 7 * 7, 512)  # Adjusted input size here
        self.output_layer = nn.Linear(512, 4)

    def forward(self, frame):
        x = torch.relu(self.conv1(frame))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.relu(self.dense(x))
        buttons = self.output_layer(x)
        return buttons

# Corrected indentation for batch size and input shape definition
batch_size = 32
input_shape = (4, 84, 84)  # Assuming 84x84 input size based on Atari games

# Generate random data as a placeholder for the batch input
sample_batch = torch.rand(batch_size, *input_shape)

# Create an instance of the DQNCNN model
model = DQNCNN()

# Pass the sample batch through the model
output = model(sample_batch)

# Ensure the output shape is as expected
print(output.shape)