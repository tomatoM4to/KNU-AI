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
        self.grid_size = int(np.sqrt(n_observations))
        if self.grid_size * self.grid_size != n_observations:
            raise ValueError("n_observations must be a perfect square")

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        conv_out_size = self.grid_size * self.grid_size * 64

        self.fc1 = nn.Linear(conv_out_size, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.grid_size, self.grid_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

mapping = {
    'E':0.0,
    'W':0.1,
    'AL':0.3,
    'AR':0.4,
    'AU':0.5,
    'AD':0.6,
    'B':0.7,
    'H':0.8,
    'K':1.0,
}

def reset_state(env):
    state, hp = env['grid'], env['hit_points']
    vectorized_mapping = np.vectorize(mapping.get)
    state = vectorized_mapping(state)
    state = state.astype(np.float32)

    return state.flatten(), hp

def calculate_reward(before_bee, after_bee, hp, e):
    r = before_bee - after_bee
    r *= hp / 100.0 * e
    return r * 3


def record_history(history):
    plt.plot(range(len(history)), history, color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards per Episode")
    plt.savefig("dqn-reward-history.png")
    print('Complete')
