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


import torch
import torch.nn as nn
import torch.nn.functional as F


class DeeperDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DeeperDQN, self).__init__()
        # 더 깊은 네트워크 구성
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 128)
        self.layer5 = nn.Linear(128, 64)
        self.layer6 = nn.Linear(64, n_actions)

        # 드롭아웃 설정
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        return self.layer6(x)


def parse_state(state: list, target: list):
    return np.array(
        [state[0][0], state[0][1], state[0][-1], target[0], target[1], target[-1]]
    )


def calculate_reward(pre_state: list, state: list) -> Tuple[float, bool]:
    x1_prev, y1_prev, sin1_prev, _, _, _ = pre_state
    x1, y1, sin1, x2, y2, sin2 = state

    # 보상 초기화
    reward = 0.0

    # x1이 x2에 1만큼 가까워지면 0.1 보상 추가
    if abs(x1 - x2) < abs(x1_prev - x2):
        reward += 0.1

    # x1이 x2에 10 이내로 도달했을 때 종료 조건
    if abs(x1 - x2) <= 10:
        reward += 1.0
        return reward, True

    return reward, False
