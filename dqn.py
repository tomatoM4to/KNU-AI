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
        self.feature_layer = nn.Sequential(
            nn.Linear(n_observations, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()
        )
        # 가치 함수 스트림
        self.value_stream = nn.Linear(128, 1)
        # 어드밴티지 스트림
        self.advantage_stream = nn.Linear(128, n_actions)

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        return values + advantages - advantages.mean(dim=1, keepdim=True)


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
    if abs(x2 - x1) < abs(x2 - x1_prev):
        reward += 0.1

    # x1이 x2에 10 이내로 도달했을 때 종료 조건
    if abs(x2 - x1) <= 10:
        reward += 1.0
        return reward, True

    return reward, False
