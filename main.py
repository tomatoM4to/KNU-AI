from knu_rl_env.road_hog import make_road_hog, RoadHogAgent, evaluate, run_manual
from dqn import DQN, ReplayMemory, Transition, parse_state, calculate_reward
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from typing import Any, Dict
import matplotlib.pyplot as plt
import math

# from eps import AdaptiveEpsilonGreedy
# from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
LR = 1e-4

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

steps_done = 0

ENV = make_road_hog(show_screen=False)
observation: Dict[str, Any] = ENV.reset()[0]
is_on_load: bool = observation["is_on_load"]
is_crashed: bool = observation["is_crashed"]
o_time: int = observation["time"]

# 환경 초기화
state: list = parse_state(observation["observation"], observation["goal_spot"])
n_state: int = len(state)
action_space = np.array([3, 4, 5])
n_actions: int = len(action_space)

# 입구 찾기 네트워크
policy_net = DQN(n_state, n_actions).to(device)

# 주차 네트워크
parking_policy_net = DQN(n_state, n_actions).to(device)
parking_target_net = DQN(n_state, n_actions).to(device)
parking_target_net.load_state_dict(parking_policy_net.state_dict())

parking_optimizer = optim.AdamW(parking_policy_net.parameters(), lr=LR, amsgrad=True)
parking_memory = ReplayMemory(50000)


class RoadHogRLAgent(RoadHogAgent):
    def __init__(self):
        super().__init__()
        self.init_action_space = [7, 7, 7, 3, 3, 7, 7, 7, 3, 3, 7, 7, 7]
        self.idx = -1
        self.action_box = []
        self.is_finder = False
        self.X = 0
        self.Y = 0

    def reset(self):
        self.idx = -1
        self.init_action_space = [7, 7, 7, 3, 3, 7, 7, 7, 3, 3, 7, 7, 7]

    def act(self, state):
        if self.init_action_space:
            return self.init_action_space.pop(0)

        if self.action_box:
            a = self.action_box.pop(0)
            return action_space[a]
        state = parse_state(state["observation"], state["goal_spot"])
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = policy_net(state).max(1).indices.view(1, 1)
        self.action_box.append(action.item())

        return action_space[action.item()]

    def policy_net_act(self, state):
        if self.init_action_space:
            return self.init_action_space.pop(0)

        state = parse_state(state["observation"], state["goal_spot"])
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = policy_net(state).max(1).indices.view(1, 1)

        return action_space[action.item()]

    def train_act(self, state: torch.Tensor) -> torch.Tensor:
        global steps_done

        sample = np.random.random()

        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * steps_done / EPS_DECAY
        )

        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return parking_policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[np.random.randint(3)]], device=device, dtype=torch.long
            )

    def optimize_model(self):

        if len(parking_memory) < BATCH_SIZE:
            return
        transitions = parking_memory.sample(BATCH_SIZE)

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

        state_action_values = parking_policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)

        with torch.no_grad():
            next_state_values[non_final_mask] = (
                parking_target_net(non_final_next_states).max(1).values
            )

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        parking_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(parking_policy_net.parameters(), 100)
        parking_optimizer.step()

    def save(self, file_path="parking_policy_net.pth"):
        torch.save(
            {
                "parking_policy_net_state_dict": parking_policy_net.state_dict(),
                "steps_done": steps_done,
            },
            file_path,
        )
        print(f"Model saved to {file_path}")

    def load_policy_net(self, file_path="policy_net.pth"):
        checkpoint = torch.load(file_path, map_location=device)
        policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        print(f"Model loaded from {file_path}")

    def load_parking_net(self, file_path="parking_policy_net.pth"):
        checkpoint = torch.load(file_path, map_location=device, weights_only=True)
        parking_policy_net.load_state_dict(checkpoint["parking_policy_net_state_dict"])
        print(f"Model loaded from {file_path}")


action_history = []


def skip_step(
    action: int,
):
    done = False
    action_space.append(action)
    for _ in range(2):
        observation, reward, terminated, truncated, _ = ENV.step(action)
        done = terminated or truncated
        if done:
            break
    return observation, done


rewards_history = []


def train(agent: RoadHogRLAgent):
    observation: Dict[str, Any]
    done: bool
    goal: bool
    state: list
    pre_state: list
    state_t: torch.Tensor
    next_state: torch.Tensor | None
    reward: int
    reward_t: torch.Tensor
    action: torch.Tensor

    num_episodes = 600

    # 클리어 조건 임계값 설정
    distance_threshold = 0.1
    sin_threshold = 0.1
    for episode in range(num_episodes):
        agent.reset()
        observation, _ = ENV.reset()

        episode_reward = 0.0
        goal = False
        while 1:
            # 객체가 주차장에 진입
            if observation["observation"][0][0] <= -55:
                action = agent.policy_net_act(observation)
                observation, done = skip_step(action)
                pre_state = parse_state(
                    observation["observation"], observation["goal_spot"]
                )
                state_t = torch.tensor(
                    pre_state, dtype=torch.float32, device=device
                ).unsqueeze(0)

                agent.X = pre_state[0]
                agent.Y = pre_state[1]
                continue

            action = agent.train_act(state_t)
            observation, done1 = skip_step(action.item())
            state = parse_state(observation["observation"], observation["goal_spot"])
            reward, done2 = calculate_reward(pre_state, state, (agent.X, agent.Y))
            pre_state = state

            done = done1 or done2

            # 클리어 확인
            if done:
                x, y, sin = state[0], state[1], state[2]
                target_x, target_y, target_sin = state[3], state[4], state[5]

                # 거리와 sin 차이 계산
                distance_to_target = ((x - target_x) ** 2 + (y - target_y) ** 2) ** 0.5
                sin_difference = abs(sin - target_sin)
                if (
                    distance_to_target < distance_threshold
                    and sin_difference < sin_threshold
                ):
                    goal = True
                    reward = 50.0

            episode_reward += reward
            reward_t = torch.tensor([reward], device=device)

            # 다음 상태를 위한 처리
            if done:
                next_state = None
            else:
                next_state = torch.tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # 메모리에 저장
            parking_memory.push(state_t, action, next_state, reward_t)

            # 다음 상태로 이동
            state_t = next_state

            # 최적화
            agent.optimize_model()

            # 목표 네트워크 가중치 소프트 업데이트
            parking_target_net_state_dict = parking_target_net.state_dict()
            parking_policy_net_state_dict = parking_policy_net.state_dict()
            for key in parking_policy_net_state_dict:
                parking_target_net_state_dict[key] = parking_policy_net_state_dict[
                    key
                ] * TAU + parking_target_net_state_dict[key] * (1 - TAU)
            parking_target_net.load_state_dict(parking_target_net_state_dict)

            if done:
                break

        rewards_history.append(episode_reward)
        if episode % 10 == 0:
            print(f"episode: {episode} reward: {episode_reward}")
            print(f"agent x: {state[0]} agent y: {state[1]}")
            print(action_history)
            print()
        action_history.clear()


if __name__ == "__main__":
    agent = RoadHogRLAgent()
    agent.load_policy_net()
    train(agent)
    agent.save()

    # evaluate(agent)

    # run_manual()

    print("Complete")
    plt.plot(range(len(rewards_history)), rewards_history, color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards per Episode")
    plt.savefig("dqn-reward-history.png")
