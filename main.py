import knu_rl_env.grid_survivor as knu
import random
import math

import torch
import torch.optim as optim
import torch.nn as nn

from dqn import DQN, ReplayMemory, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = knu.make_grid_survivor(show_screen=True)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

class GridSurvivorRLAgent(knu.GridSurvivorAgent):
    def act(self, state):
        '''
        다음 중 하나를 반환해야 합니다:
        - GridSurvivorAgent.ACTION_LEFT
        - GridSurvivorAgent.ACTION_RIGHT
        - GridSurvivorAgent.ACTION_FORWARD
        '''
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()


def train():
    done = False
    while not done:
        action = int(input())
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated


if __name__ == '__main__':
    agent = '''여러분이 정의하고 학습시킨 에이전트를 불러오는 코드를 넣으세요.'''
    # knu.evaluate(agent)
    train()