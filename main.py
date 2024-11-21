import knu_rl_env.grid_survivor as knu
import torch

'''
knu_rl_env.grid_adventure.GridAdventureAgent을 상속해서
자신만의 에이전트를 구현하세요.
'''
class GridSurvivorRLAgent(knu.GridSurvivorAgent):
    def act(self, state):
        '''
        다음 중 하나를 반환해야 합니다:
        - GridSurvivorAgent.ACTION_LEFT
        - GridSurvivorAgent.ACTION_RIGHT
        - GridSurvivorAgent.ACTION_FORWARD
        '''
        pass

'''
에이전트를 훈련하는 코드를 구현하세요.
'''
def train():
    '''
    Grid Survivor 환경은 다음과 같이 생성할 수 있습니다.
    '''
    env = knu.make_grid_suvivor(
        show_screen=True # or, False
    )
    '''
    여기서부터는 이 환경에 대해서 에이전트를 훈련시키는 코드가 필요합니다.
    '''

if __name__ == '__main__':
    agent = '''여러분이 정의하고 학습시킨 에이전트를 불러오는 코드를 넣으세요.'''
    # knu.evaluate(agent)
    t = torch.tensor([1, 2, 3])
    print(t)