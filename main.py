from knu_rl_env.road_hog import make_road_hog, RoadHogAgent, evaluate
from dqn import DQN, ReplayMemory, Transition
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import math
import time
from typing import Any, Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

env: Dict[str, Any] = make_road_hog(show_screen=False)
observation: list = env["observation"]
goal_spot: list = env["goal_spot"]
is_on_load: bool = env["is_on_load"]
is_crashed: bool = env["is_crashed"]
o_time: int = env["time"]


n_observations: int = 9
n_actions: int = 9

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


class RoadHogRLAgent(RoadHogAgent):
    def act(self, state):
        print(state)
        time.sleep(1)
        return np.random.randint(9)

    def train_act(self, state):
        sample = np.random.random()

        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * steps_done / EPS_DECAY
        )

        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[np.random.randint(9)]], device=device, dtype=torch.long
            )

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )

        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)

        with torch.no_grad():
            next_state_values[non_final_mask] = (
                target_net(non_final_next_states).max(1).values
            )

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()


def train():
    agent = RoadHogRLAgent()
    for episode in range(1000):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0
        while 1:
            action = agent.train_act(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward
            reward = torch.tensor([reward], device=device)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # 메모리에 변이 저장
            memory.push(state, action, next_state, reward)

            # 다음 상태로 이동
            state = next_state

            # (정책 네트워크에서) 최적화 한단계 수행
            agent.optimize_model()

            # 목표 네트워크의 가중치를 소프트 업데이트
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if terminated or truncated:
                break


if __name__ == "__main__":
    # agent = RoadHogRLAgent()
    # evaluate(agent)
    train()
