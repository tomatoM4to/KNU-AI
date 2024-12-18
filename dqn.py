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
    pre_x, pre_y, pre_sin = pre_state[0], pre_state[1], pre_state[2]
    x, y, sin = state[0], state[1], state[2]
    target_x, target_y, target_sin = state[3], state[4], state[5]

    initial_x, initial_y = initialAgentLoc
    boundaryX1 = initial_x - 1
    boundaryX2 = target_x + 3
    boundaryY1 = target_y - 3
    boundaryY2 = initial_y + 3

    # 경계값 벗어나면 즉시 종료
    if x < boundaryX1 or x > boundaryX2 or y < boundaryY1 or y > boundaryY2:
        return -1.0, True

    # 이전 step과 현재 step에서 x, y와 목표의 차이
    pre_x_diff = abs(pre_x - target_x)
    current_x_diff = abs(x - target_x)
    pre_y_diff = abs(pre_y - target_y)
    current_y_diff = abs(y - target_y)

    # 이전 step과 현재 step에서 사인값과 목표의 차이
    pre_sin_diff = abs(pre_sin - target_sin)
    current_sin_diff = abs(sin - target_sin)

    # 보상
    reward = 0.0

    # X축 개선 시 보상
    if current_x_diff < pre_x_diff:
        reward += 0.5

    # Y축 개선 시 보상
    if current_y_diff < pre_y_diff:
        reward += 0.5

    # X, Y 개선이 있었다면 사인값 개선 여부 체크
    if reward > 0.0 and current_sin_diff < pre_sin_diff:
        reward += 0.5

    if current_sin_diff < 0.1:
        reward += 0.5

    return reward, False
