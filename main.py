import knu_rl_env.grid_survivor as knu
import random
import math
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from dqn import DQN, ReplayMemory, Transition, reset_state, record_history, calculate_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = knu.make_grid_survivor(show_screen=False)

class GridSurvivorRLAgent(knu.GridSurvivorAgent):
    def __init__(self):
        super(GridSurvivorRLAgent, self).__init__()

        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4

        self.n_actions = 3
        self.state = reset_state(env.reset()[0])[0]
        self.n_observations = len(self.state)

        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    def act(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[np.random.randint(3)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def get_epsilon(self):
        return self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)

    def reset(self):
        self.steps_done = 0

def train(episodes):
    agent = GridSurvivorRLAgent()
    history = []
    for e in range(episodes):
        state, hp = reset_state(env.reset()[0])
        bee = np.count_nonzero(state == 7) # 꿀벌의 개수
        _bee = bee
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        record_reward = 0
        episode = 1.0
        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())

            next_state, hp = reset_state(next_state)
            _bee = np.count_nonzero(next_state == 7)
            reward = calculate_reward(_bee, bee, hp, episode)
            episode *= 0.95
            bee = np.count_nonzero(next_state == 7)
            # 기록
            record_reward += reward

            reward = torch.tensor([reward], device=device)

            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            agent.memory.push(state, action, next_state, reward)

            state = next_state

            agent.optimize_model()

            # 네트워크 업데이트
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * agent.TAU + target_net_state_dict[key] * (1 - agent.TAU)
            agent.target_net.load_state_dict(target_net_state_dict)

            if done:
                break
        if e % 50 == 0:
            print(f'Episode {e} - Reward: {record_reward}, Bee: {bee}')
        history.append(_bee)

    return history



if __name__ == '__main__':
    history = train(600)
    record_history(history)

    # agent = '''여러분이 정의하고 학습시킨 에이전트를 불러오는 코드를 넣으세요.'''
    # knu.evaluate(agent)