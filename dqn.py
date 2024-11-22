import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # Grid를 2D로 재구성
        self.grid_size = int(np.sqrt(n_observations))

        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Calculate the size after convolutions
        conv_out_size = self.grid_size * self.grid_size * 64

        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        batch_size = x.size(0)
        # Reshape input to 2D grid
        x = x.view(batch_size, 1, self.grid_size, self.grid_size)

        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten
        x = x.view(batch_size, -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

mapping = {
    'E':0,
    'W':1,
    'AL':3,
    'AR':4,
    'AU':5,
    'AD':6,
    'B':7,
    'H':8,
    'K':9,
}

def reset_state(env):
    state, hp = env['grid'], env['hit_points']
    vectorized_mapping = np.vectorize(mapping.get)
    state = vectorized_mapping(state)
    return state.flatten(), hp

def calculate_reward(_bee, bee, hp, e):
    if _bee == bee:
        return 0.0
    r = 1.0
    r *= hp / 100.0 * e
    return r


def record_history(history):
    plt.plot(range(len(history)), history, color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards per Episode")
    plt.savefig("dqn-reward-history.png")
    print('Complete')
