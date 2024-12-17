import knu_rl_env as knu
from knu_rl_env.road_hog import make_road_hog, RoadHogAgent, evaluate
from dqn import DQN, ReplayMemory, Transition, parse_state, calculate_reward
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from typing import Any, Dict
import matplotlib.pyplot as plt
from eps import AdaptiveEpsilonGreedy
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
LR = 1e-4

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

policy_net = DQN(n_state, n_actions).to(device)
target_net = DQN(n_state, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(50000)


class RoadHogRLAgent(RoadHogAgent):
    def __init__(self):
        super().__init__()
        self.init_action_space = np.array([7, 7, 7, 3, 3, 7, 7, 7, 3, 3, 7, 7, 7])
        self.idx = -1
        self.loc = np.array([0, 0, 0])

        self.epsilon = AdaptiveEpsilonGreedy()

    def reset(self):
        self.idx = -1

    def env_reset(self):
        for i in self.init_action_space:
            ENV.step(i)

    def act(self, state):
        return 4

    def train_act(self, state: torch.Tensor) -> torch.Tensor:
        global steps_done

        sample = np.random.random()

        eps_threshold = self.epsilon.epsilon

        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[np.random.randint(3)]], device=device, dtype=torch.long
            )

    def optimize_model(self):

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


def skip_step(
    action: torch.Tensor,
):
    done = False
    a = action_space[action.item()]
    for _ in range(3):
        observation, reward, terminated, truncated, _ = ENV.step(a)
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
    window_size = 50
    recent_rewards = deque([], maxlen=window_size)
    recent_successes = deque([], maxlen=window_size)

    for episode in range(num_episodes):
        agent.reset()
        observation, _ = ENV.reset()
        pre_state = parse_state(observation["observation"], observation["goal_spot"])
        state_t = torch.tensor(pre_state, dtype=torch.float32, device=device).unsqueeze(
            0
        )

        episode_reward = 0.0
        goal = False

        agent.env_reset()
        while 1:
            action = agent.train_act(state_t)
            observation, done = skip_step(action)
            state = parse_state(observation["observation"], observation["goal_spot"])
            reward, goal = calculate_reward(pre_state, state)
            pre_state = state

            # 보상 관련 처리
            if done:
                reward = -1.0
            elif not observation["is_on_load"]:
                reward = -1.0
                done = True

            if goal:
                done = True

            episode_reward += reward
            reward_t = torch.tensor([reward], device=device)

            # 다음 상태를 위한 처리
            if done:
                next_state = None
            else:
                next_state = torch.tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # 메모리에 변이 저장
            memory.push(state_t, action, next_state, reward_t)

            # 다음 상태로 이동
            state_t = next_state

            # (정책 네트워크에서) 최적화 한단계 수행
            agent.optimize_model()

            # 목표 네트워크의 가중치를 소프트 업데이트
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                break

        # 에피소드 결과 기록
        recent_rewards.append(episode_reward)
        recent_successes.append(1.0 if goal else 0.0)

        # 일정 기간(윈도우) 평균으로 epsilon 업데이트
        if len(recent_rewards) == window_size:
            avg_reward = np.mean(recent_rewards)
            success_rate = np.mean(recent_successes)
            agent.epsilon.update_epsilon(avg_reward, success_rate)

        rewards_history.append(episode_reward)
        if episode % 10 == 0:
            print(
                f"episode: {episode} reward: {episode_reward} epsilon: {agent.epsilon.epsilon}"
            )
            print(f"agent x: {state[0]} goal x: {state[3]}")


if __name__ == "__main__":
    agent = RoadHogRLAgent()
    # evaluate(agent)
    train(agent)
    # knu.road_hog.run_manual()
    print("Complete")
    plt.plot(range(len(rewards_history)), rewards_history, color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards per Episode")
    plt.savefig("dqn-reward-history.png")
