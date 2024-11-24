import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 경험 리플레이를 위한 Transition 정의
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# 경험 리플레이 메모리 클래스
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

# DQN 모델 정의
class DQN(nn.Module):
    def __init__(self, height, width, n_actions, input_channels):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # 출력 크기 계산 함수 정의
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + 2*padding - kernel_size) // stride + 1

        def pool2d_size_out(size, kernel_size=2, stride=2):
            return (size - kernel_size) // stride + 1

        # 높이와 너비 계산
        h = height
        w = width

        h = conv2d_size_out(h, kernel_size=5, stride=1, padding=2)
        h = pool2d_size_out(h, kernel_size=2, stride=2)
        h = conv2d_size_out(h, kernel_size=3, stride=1, padding=1)
        h = pool2d_size_out(h, kernel_size=2, stride=2)
        h = conv2d_size_out(h, kernel_size=3, stride=1, padding=1)

        w = conv2d_size_out(w, kernel_size=5, stride=1, padding=2)
        w = pool2d_size_out(w, kernel_size=2, stride=2)
        w = conv2d_size_out(w, kernel_size=3, stride=1, padding=1)
        w = pool2d_size_out(w, kernel_size=2, stride=2)
        w = conv2d_size_out(w, kernel_size=3, stride=1, padding=1)

        conv_out_size = h * w * 64

        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

        # 가중치 초기화
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))

        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 상태 매핑을 위한 키 리스트
mapping = ['E', 'W', 'AL', 'AR', 'AU', 'AD', 'B', 'H', 'K']

# 상태를 채널별로 분리하는 함수
def reset_state(env):
    state_grid, hp = env['grid'], env['hit_points']
    channels = []
    for key in mapping:
        channel = (state_grid == key).astype(np.float32)
        channels.append(channel)
    state_tensor = np.stack(channels, axis=0)  # (채널 수, 높이, 너비)
    return state_tensor, hp

# 보상 함수 수정
def calculate_reward(before_bee, after_bee, hp, walk):
    if before_bee == after_bee:
        return -0.1 - (walk / 600.0)

    reward = 0.0
    if after_bee < before_bee:
        # 꿀벌을 구하면 +10점
        reward += 10.0

    # 남은 체력에 보상 감소율 적용
    reward += (hp / 100.0) * 5.0

    return reward - (walk / 1200.0)


def record_history(history):
    plt.plot(range(len(history)), history, color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards per Episode")
    plt.savefig("dqn-reward-history.png")
    print('Complete')
