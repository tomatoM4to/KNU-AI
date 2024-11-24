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


class FrameProcessor:
    def __init__(self, n_frames=4, frame_skip=1):
        self.n_frames = n_frames
        self.frame_skip = frame_skip
        self.frame_diffs = None # deque 객체로 초기화됨
        self.last_frame = None # 첫 프레임으로 초기화됨
        self.skip_counter = 0 # 프레임 스킵 카운터
        self.accumulated_reward = 0 # 누적 보상

    def reset(self, initial_frame):
        """
        1. 마지막 프레임을 초기 프레임으로 설정
        2. frame_diffs를 0으로 채워진 n_frames 크기의 deque로 초기화
        3. 스킵 카운터 초기화
        4. 누적 보상 초기화
        """
        self.last_frame = initial_frame
        self.frame_diffs = deque([np.zeros_like(initial_frame)] * self.n_frames, maxlen=self.n_frames)
        self.skip_counter = 0
        self.accumulated_reward = 0

    def process_frame(self, new_frame, reward):
        self.skip_counter += 1
        self.accumulated_reward += reward

        """
        1. frame_skip에 도달할시:
        2. 현재 프레임과 마지막 프레임의 차이 계산
        3. frame_diffs에 차이 추가
        4. 마지막 프레임을 현재 프레임으로 설정
        5. 스킵 카운터, 누적 보상 초기화
        """
        if self.skip_counter >= self.frame_skip:
            # new_frame이 None이면 0으로 채워진 프레임 차이 사용
            if new_frame is None:
                frame_diff = np.zeros_like(self.last_frame)
            else:
                frame_diff = new_frame

            self.frame_diffs.append(frame_diff)

            total_reward = self.accumulated_reward # 누적 보상을 반환하기 위한 변수

            self.last_frame = new_frame
            self.skip_counter = 0
            self.accumulated_reward = 0
            return True, total_reward

        return False, 0

    def get_state(self):
        return np.stack(self.frame_diffs, axis=0)

    def get_state_tensor(self):
        state = self.get_state()
        return torch.FloatTensor(state).unsqueeze(0).to(device)

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
    def __init__(self, grid_size, n_actions, n_frames):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(n_frames, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Calculate the size after convolutions and pooling
        conv_out_size = grid_size // 4  # After two 2x2 pooling layers
        conv_out_size = conv_out_size * conv_out_size * 64

        self.ln1 = nn.Linear(conv_out_size, 512)

        self.ln2 = nn.Linear(512, n_actions)

        # Initialize weights
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
        x = F.relu(self.ln1(x))
        x = self.ln2(x)
        return x

mapping = {
    'E': 0.0,
    'W': 0.1,
    'AL': 0.2,
    'AR': 0.3,
    'AU': 0.4,
    'AD': 0.5,
    'B': 0.6,
    'H': 0.9,
    'K': 1.0,
}

# String type to float type
def reset_state(env):
    state, hp = env['grid'], env['hit_points']
    vectorized_mapping = np.vectorize(mapping.get)
    state = vectorized_mapping(state)
    state = state.astype(np.float32)
    return state, hp

def calculate_reward(before_bee, after_bee, hp, game_steps):
    # 꿀벌이 먹지 않았다면 0.1점
    if before_bee == after_bee:
        return 0.1

    # 꿀벌을 먹었다면 기본 보상 10점
    reward = 10

    # 체력 보너스 (0.5 ~ 1.0)
    survival_bonus = hp / 200.0

    # 최소 보상 3점
    return max(reward * survival_bonus * game_steps, 3)


def record_history(history):
    plt.plot(range(len(history)), history, color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards per Episode")
    plt.savefig("dqn-reward-history.png")
    print('Complete')