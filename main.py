from knu_rl_env.road_hog import make_road_hog, RoadHogAgent, evaluate
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
knu_rl_env.road_hog.RoadHogAgent을 상속해서
자신만의 에이전트를 구현하세요.
'''
class RoadHogRLAgent(RoadHogAgent):
    def act(self, state):
        '''
        Return value is one of actions following:
        - RoadHogAgent.FORWARD_ACCEL_RIGHT
        - RoadHogAgent.FORWARD_ACCEL_NEUTRAL
        - RoadHogAgent.FORWARD_ACCEL_LEFT
        - RoadHogAgent.NON_ACCEL_RIGHT
        - RoadHogAgent.NON_ACCEL_NEUTRAL
        - RoadHogAgent.NON_ACCEL_LEFT
        - RoadHogAgent.BACKWARD_ACCEL_RIGHT
        - RoadHogAgent.BACKWARD_ACCEL_NEUTRAL
        - RoadHogAgent.BACKWARD_ACCEL_LEFT
        '''
        return np.random.randint(9)


'''
에이전트를 훈련하는 코드를 구현하세요.
'''
def train():
    '''
    Road Hog 환경은 다음과 같이 생성할 수 있습니다.
    '''
    env = make_road_hog(
        show_screen=True # or, False
    )
    '''
    여기서부터는 이 환경에 대해서 에이전트를 훈련시키는 코드가 필요합니다.
    '''

if __name__ == '__main__':
    agent = RoadHogRLAgent()
    evaluate(agent)