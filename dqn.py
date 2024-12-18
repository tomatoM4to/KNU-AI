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

    # 초기 위치 추출
    initial_x, initial_y = initialAgentLoc
    boundaryX1 = initial_x - 1  # 초기 위치 기준 x 경계
    boundaryX2 = target_x + 3  # 목표 위치 기준 x 경계
    boundaryY1 = target_y - 3  # 목표 위치 기준 y 경계
    boundaryY2 = initial_y + 2  # 초기 위치 기준 y 경계

    # 이전과 현재의 목표와의 거리 계산
    pre_distance_to_target = ((pre_x - target_x) ** 2 + (pre_y - target_y) ** 2) ** 0.5
    current_distance_to_target = ((x - target_x) ** 2 + (y - target_y) ** 2) ** 0.5

    # 목표 기울기(sin) 차이 계산
    sin_difference = abs(sin - target_sin)

    # 보상 정의
    reward = 0.0

    # 경계값 벗어나는 경우
    if x < boundaryX1 or x > boundaryX2 or y < boundaryY1 or y > boundaryY2:
        return -1.0, True

    # 목표에 가까워지면 보상
    if current_distance_to_target < pre_distance_to_target:
        reward += 0.5

    # 기울기 차이가 작을수록 보상 증가
    reward += max(1.0 - sin_difference, 0)

    return reward, False
