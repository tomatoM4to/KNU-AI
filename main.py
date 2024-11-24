import knu_rl_env.grid_survivor as knu
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from dqn import DQN, ReplayMemory, Transition, reset_state, calculate_reward, FrameProcessor, record_history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = knu.make_grid_survivor(show_screen=False)

class GridSurvivorRLAgent:
    def __init__(self):
        # Hyper parameters
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.TAU = 0.005
        self.LR = 1e-4
        self.N_FRAMES = 4
        self.FRAME_SKIP = 1

        # Epsilon greedy parameters
        self.eps_start = 1.0
        self.EPS_END = 0.05

        # Environment parameters
        N_ACTIONS = 3
        STATE, _ = reset_state(env.reset()[0])
        N_OBSERVATIONS = len(STATE)

        # Initialize frame processor
        self.frame_processor = FrameProcessor(self.N_FRAMES, self.FRAME_SKIP)
        self.frame_processor.reset(STATE)

        # Initialize networks
        self.policy_net = DQN(N_OBSERVATIONS, N_ACTIONS, self.N_FRAMES).to(device)
        self.target_net = DQN(N_OBSERVATIONS, N_ACTIONS, self.N_FRAMES).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initialize optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(),
                                   lr=self.LR,
                                   amsgrad=True,
                                   weight_decay=1e-5)  # Added weight decay

        # Initialize memory
        self.memory = ReplayMemory(100000)  # Increased memory size

        # Initialize steps
        self.steps_done = 0

    def discountEPS(self, episode):
        speed = 0.1 * (episode / 4000) ** 2
        self.eps_start -= speed * (self.eps_start - self.EPS_END) / 20
        self.eps_start = max(self.eps_start, self.EPS_END)


    def getEPS(self):
        return self.eps_start

    def act_train(self, state):
        e = self.getEPS()

        self.steps_done += 1

        if np.random.rand() > e:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[np.random.randint(3)]], device=device, dtype=torch.long)

    def reset_to_test(self):
        try:
            self.load('model.pth')
            self.policy_net.eval()
            self.target_net.eval()
            print('Model loaded successfully')
        except:
            print('Model not found')
            raise


    def act(self, state):
        # print(state)
        state, _ = reset_state(state)
        # state를 frame processor로 처리
        self.frame_processor.process_frame(state, 0)  # reward는 테스트에서 중요하지    않으므로 0
        current_state = self.frame_processor.get_state_tensor()

        with torch.no_grad():
            action = self.policy_net(current_state).max(1).indices.view(1, 1)
        return action.item()


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
        checkpoint = torch.load(filename, weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']

def train(episodes):
    agent = GridSurvivorRLAgent()
    history = []

    for e in range(episodes + 1):
        # 환경, game_steps, 초기 꿀벌 개수 초기화
        state, hp = reset_state(env.reset()[0])
        before_bee = np.count_nonzero(state == 0.6)
        game_steps = 1

        # frame processor 초기화, 현재 상태 저장
        agent.frame_processor.reset(state)
        current_state = agent.frame_processor.get_state_tensor()

        # 기록용 변수 초기화
        episode_reward = 0
        walk = 0
        while True:
            # 에피소드 기반 행동 선택
            action = agent.act_train(current_state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())

            # String 타입 -> Float 타입
            next_state, hp = reset_state(next_state)

            # 보상 관련 프로세스
            after_bee = np.count_nonzero(next_state == 0.6)
            reward = calculate_reward(before_bee, after_bee, hp, game_steps)
            if terminated or truncated: # 종료 시 -10점
                reward = -50
                next_state = None
            if after_bee == 0: # 클리어
                reward = 100
            game_steps *= 0.9999
            before_bee = after_bee
            episode_reward += reward
            walk += 1


            # 현재 상태, reward를 프레임 프로세서에 전달, return-type: [bool, accumulated_reward]
            processed, total_reward = agent.frame_processor.process_frame(next_state, reward)
            if processed:
                next_state_tensor = None if next_state is None else agent.frame_processor.get_state_tensor()

                # 메모리에 저장
                agent.memory.push(current_state,
                                action,
                                next_state_tensor,
                                torch.tensor([total_reward], device=device))

                # 모델 최적화
                loss = agent.optimize_model()

                # 초기화
                current_state = next_state_tensor

            # 타겟 네트워크 업데이트
            if agent.steps_done % 500 == 0:
                agent.update_target_net()

            if terminated or truncated:
                break
        if e % 10 == 0:
            print(f'Episode {e} - Reward: {episode_reward}, bee: {after_bee}, EPS: {agent.getEPS()}, walk: {walk}')
        agent.discountEPS(e)
        history.append(episode_reward)
    agent.save()
    return history

if __name__ == '__main__':
    history = train(4500)
    record_history(history)

    # agent = GridSurvivorRLAgent()
    # agent.reset_to_test()
    # knu.evaluate(agent)