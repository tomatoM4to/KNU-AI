import knu_rl_env.grid_survivor as knu
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from dqn import DQN, ReplayMemory, Transition, reset_state, calculate_reward, record_history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = knu.make_grid_survivor(show_screen=False)

class GridSurvivorRLAgent:
    def __init__(self):
        # Hyper parameters
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.TAU = 0.005
        self.LR = 1e-4

        # Epsilon greedy parameters
        self.eps_start = 1.0
        self.EPS_END = 0.05

        # Environment parameters
        N_ACTIONS = 3
        STATE, _ = reset_state(env.reset()[0])
        input_channels, height, width = STATE.shape

        # Initialize networks
        self.policy_net = DQN(height, width, N_ACTIONS, input_channels).to(device)
        self.target_net = DQN(height, width, N_ACTIONS, input_channels).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initialize optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(),
                                   lr=self.LR,
                                   amsgrad=True,
                                   weight_decay=1e-5)  # Added weight decay

        # Initialize memory
        self.memory = ReplayMemory(50000)  # Increased memory size

        # Initialize steps
        self.steps_done = 0

    def discountEPS(self, episode):
        # 탐험률을 천천히 감소시킴
        self.eps_start -= (self.eps_start - self.EPS_END) / 1000
        self.eps_start = max(self.EPS_END, self.eps_start)

    def getEPS(self):
        return self.eps_start

    def act(self, state):
        e = self.getEPS()

        self.steps_done += 1

        if np.random.rand() > e:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[np.random.randint(3)]], device=device, dtype=torch.long)

    def optimize_model(self):
        # 배치 크기만큼의 메모리가 쌓이지 않았다면 학습하지 않음
        if len(self.memory) < self.BATCH_SIZE:
            return

        # 메모리에서 무작위로 배치 크기만큼 트랜지션 샘플링
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # next_state 샘플들의 마스크 생성 (게임이 끝나지 않은 상태들)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)),
                                    device=device, dtype=torch.bool)

        # 다음 상태들을 하나의 텐서로 결합
        non_final_next_states = torch.cat([s for s in batch.next_state
                                         if s is not None])

        # 현재 상태, 행동, 보상을 텐서로 변환
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Double DQN
        with torch.no_grad():
            # 정책 네트워크로 다음 상태에서의 최적 행동 선택
            next_action_values = self.policy_net(non_final_next_states)
            next_actions = next_action_values.max(1)[1].unsqueeze(1)

            # 타겟 네트워크로 선택된 행동의 가치 계산
            next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).squeeze(1)

        # Q(s,a) = r + γ * max Q(s',a')
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # 현재 정책 네트워크의 Q 값 계산
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Huber Loss를 사용하여 손실 계산
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 역전파 및 최적화
        self.optimizer.zero_grad() # gradient 초기화
        loss.backward() # 역전파
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) # 클리핑
        self.optimizer.step() # 파라미터 업데이트

        return loss.item()

    def update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + \
                                       target_net_state_dict[key] * (1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def save(self):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, 'model.pth')

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']

def train(episodes):
    agent = GridSurvivorRLAgent()
    history = []

    for e in range(episodes + 1):
        # 환경 초기화
        state, hp = reset_state(env.reset()[0])
        before_bee = np.count_nonzero(state[6])  # 채널 6이 꿀벌(B)에 해당
        steps = 1.0

        # 상태를 텐서로 변환
        current_state = torch.FloatTensor(state).unsqueeze(0).to(device)  # (1, 채널 수, 높이, 너비)

        # 기록용 변수 초기화
        episode_reward = 0
        walk = 1

        while True:
            # 행동 선택
            action = agent.act(current_state)
            next_state_raw, reward, terminated, truncated, _ = env.step(action.item())

            # 다음 상태 처리
            if not terminated and not truncated:
                next_state, hp = reset_state(next_state_raw)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            else:
                next_state_tensor = None

            # 보상 계산
            if next_state_tensor is not None:
                after_bee = np.count_nonzero(next_state[6])
            else:
                # 에피소드 종료 시에도 남은 꿀벌 수를 계산하기 위해 마지막 상태를 사용
                after_bee = np.count_nonzero(next_state[6])

            reward = calculate_reward(before_bee, after_bee, hp, walk)

            # 모든 꿀벌을 구했는지 확인
            if after_bee == 0 and not terminated and not truncated:
                # 모든 꿀벌을 구해서 에피소드가 종료되지 않은 경우
                reward += 50

            # 에피소드가 종료되었을 때 패널티 부여
            if (terminated or truncated) and after_bee > 0:
                reward -= 50

            before_bee = after_bee
            episode_reward += reward
            walk += 1
            steps *= 0.9999

            # 메모리에 저장
            agent.memory.push(current_state,
                              action,
                              next_state_tensor,
                              torch.tensor([reward], device=device))

            # 모델 최적화
            loss = agent.optimize_model()

            # 현재 상태 업데이트
            current_state = next_state_tensor

            # 타겟 네트워크 소프트 업데이트
            agent.update_target_net()

            if terminated or truncated:
                break


        if e % 100 == 0:
            print(f'Episode {e} - Reward: {episode_reward}, bees left: {after_bee}, EPS: {agent.getEPS()}, walk: {walk}')

        agent.discountEPS(e)

        history.append(episode_reward)

    agent.save()
    return history

if __name__ == '__main__':
    history = train(4000)
    record_history(history)