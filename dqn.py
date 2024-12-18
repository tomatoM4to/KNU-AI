import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import numpy as np
from typing import Tuple
import math

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
    x, y, sin_angle = state[0], state[1], state[2]
    target_x, target_y, target_sin = state[3], state[4], state[5]

    initial_x, initial_y = initialAgentLoc
    boundaryX1 = initial_x - 1
    boundaryX2 = target_x + 3
    boundaryY1 = target_y - 3
    boundaryY2 = initial_y + 3

    reward = 0.0
    done = False

    # 경계값 벗어나면 즉시 종료
    if x < boundaryX1 or x > boundaryX2 or y < boundaryY1 or y > boundaryY2:
        return -1.0, True

    # 거리 계산
    dist = math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)

    # 거리 구간별 보상 정의
    if dist >= 30:
        # 멀리 있을 때 (직선적으로 유지)
        if -0.3 <= sin_angle <= -0.2:
            reward += 0.2
    elif 20 <= dist < 30:
        # 중간 거리 (약간의 방향 조정 필요)
        if -0.6 <= sin_angle <= -0.4:
            reward += 0.3
    elif 10 <= dist < 20:
        # 근접 거리 (더 큰 방향 조정 필요)
        if -0.9 <= sin_angle <= -0.7:
            reward += 0.5
    elif dist < 10:
        # 매우 근접 (정확히 목표 방향)
        if -1.0 <= sin_angle <= -0.9:
            reward += 1.0
            done = True

    return reward, done
