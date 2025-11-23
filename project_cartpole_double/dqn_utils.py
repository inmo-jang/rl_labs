# dqn_utils.py
import torch
import torch.nn as nn
from collections import namedtuple, deque
import random
import torch.nn.functional as F

# Define a named tuple for storing experiences
Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward'])

# Replay Memory Class to store and sample experiences
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Randomly samples batch_size transitions from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the current size of internal memory."""
        return len(self.memory)

# DQN Model
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)