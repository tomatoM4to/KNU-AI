# 가상환경

```shell
python -m venv venv
```
```shell
source venv/Scripts/activate
```

***

# 제출 파일 생성
```shell
pip list –format=freeze > requirements.txt
```

***

# 의존성 리스트
```shell
pip freeze > pip.txt
```
```shell
pip install -r pip.txt
```

***

# Road_hog
```python
observation: [
    [-233.66608, -12.3798065, -0., 0., -0.36411196, 0.9313552],
    [-227.54099, 15.737161, 2.098288, 3.4054642, 0.524572, 0.85136604],
    [-210.45044, 28.785242, 3.8380322, 1.1267244, 0.95950806, 0.2816811],
    [-212.08769, 32.469036, 3.8198867, 1.1867877, 0.9549717, 0.29669693],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.]
]
goal_spot:  [ 1.400000e+01 -4.000000e+01  0.000000e+00  0.000000e+00  6.123234e-17  -1.000000e+00 ]
is_on_load:  True
is_crashed:  False
o_time:  0
```

* **observation**: 10 x 6 배열 반환, 첫번째 행은 현재 플레이어가 조작하는 차량(x좌표, y좌표, y축 속도, 객체가 향하는 코사인 값, 객체가 양하는 사인 값), 나머진 그 외 객체들(다른차략, 벽, 목표지점)
* **goal_spot**: 목표 지점에 대한 정보
* **is_on_load**: 현재 플레이어의 차량이 차선을 위반할 시 False 반환
* **is_crashed**: 현재 플레이어의 차량이 다른 차량 또는 객체와 중돌했을시 True 반환
* **time**: 현재까지 지나간 시간

## Action Space
* knu_rl_env.RoadHogAgent.FORWARD_ACCEL_RIGHT
* `knu_rl_env.RoadHogAgent.FORWARD_ACCEL_NEUTRAL`: 앞으로 가속, 1
* knu_rl_env.RoadHogAgent.FORWARD_ACCEL_LEFT
* `knu_rl_env.RoadHogAgent.NON_ACCEL_RIGHT`: 우측, 3, 제자리에서 회전은 안됨
* knu_rl_env.RoadHogAgent.NON_ACCEL_NEUTRAL
* `knu_rl_env.RoadHogAgent.NON_ACCEL_LEFT`: 좌측, 5, 제자리에서 회전은 안됨
* knu_rl_env.RoadHogAgent.BACKWARD_ACCEL_RIGHT
* `knu_rl_env.RoadHogAgent.BACKWARD_ACCEL_NEUTRAL`: 감속, 7
* knu_rl_env.RoadHogAgent.BACKWARD_ACCEL_LEFT


## Grade
* 주차에 성공한 횟수가 많을수록 좋음
* 주차에 성공한 횟수가 동일하다면, 운전 시간 + 벌점 시간의 합이 적을수록 좋음
* 주차에 한 번도 성공하지 못했다면, 남은 거리의 합이 적을 수록 좋음
