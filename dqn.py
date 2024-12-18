import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import numpy as np
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


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
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def parse_state(state: list, target: list):
    return np.array(
        [state[0][0], state[0][1], state[0][-1], target[0], target[1], target[-1]]
    )


def calculate_reward(
    pre_state: list, state: list, initialAgentLoc: Tuple
) -> Tuple[float, bool]:
    pre_x, pre_y = pre_state[0], pre_state[1]
    x, y = state[0], state[1]
    target_x, target_y = state[3], state[4]

    initial_x, initial_y = initialAgentLoc
    boundaryX1 = initial_x - 1
    boundaryX2 = target_x + 3
    boundaryY1 = target_y - 3
    boundaryY2 = initial_y + 2

    # 이전 거리와 현재 거리
    pre_distance_to_target = ((pre_x - target_x) ** 2 + (pre_y - target_y) ** 2) ** 0.5
    current_distance_to_target = ((x - target_x) ** 2 + (y - target_y) ** 2) ** 0.5

    # 보상
    reward = 0.0

    # 경계값 벗어나면 즉시 종료
    if x < boundaryX1 or x > boundaryX2 or y < boundaryY1 or y > boundaryY2:
        return -5.0, True

    # 거리 차이 기반 보상
    distance_diff = pre_distance_to_target - current_distance_to_target
    reward += distance_diff * 0.5  # 거리에 비례한 보상 스케일링

    return reward, False
