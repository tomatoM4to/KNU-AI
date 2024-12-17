import math
import numpy as np
import torch


class AdaptiveEpsilonGreedy:
    def __init__(
        self,
        eps_max=1.0,
        eps_min=0.1,
        reward_improve_threshold=0.05,
        success_threshold=0.8,
        smoothing_factor=0.1,
        stable_tolerance=0.01,
        patience=5,
    ):
        """
        eps_max: 초기 최대 탐색률
        eps_min: 최소 탐색률
        reward_improve_threshold: 보상 개선 판단 비율
        success_threshold: 성공률 임계치
        smoothing_factor: EMA 적용을 위한 계수(0~1)
        stable_tolerance: 정체 구간 판단 기준 (보상 변화율이 이 이하일 경우 정체)
        patience: 정체 구간 허용 횟수
        """
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.epsilon = eps_max

        self.avg_reward_ema = None
        self.success_ema = None
        self.smoothing_factor = smoothing_factor

        self.reward_improve_threshold = reward_improve_threshold
        self.success_threshold = success_threshold

        self.increase_factor = 1.01
        self.decrease_factor = 0.99

        self.stable_tolerance = stable_tolerance
        self.patience = patience
        self.stable_count = 0

    def update_epsilon(self, avg_reward, success_rate):
        # EMA 업데이트
        if self.avg_reward_ema is None:
            self.avg_reward_ema = avg_reward
            self.success_ema = success_rate
            return
        else:
            old_ema = self.avg_reward_ema
            self.avg_reward_ema = (self.smoothing_factor * avg_reward) + (
                (1 - self.smoothing_factor) * self.avg_reward_ema
            )
            self.success_ema = (self.smoothing_factor * success_rate) + (
                (1 - self.smoothing_factor) * self.success_ema
            )

        # 이전 EMA 값 대비 변화량 계산
        current_ema = self.avg_reward_ema
        current_diff = current_ema - old_ema

        improvement_condition = (
            current_diff > self.reward_improve_threshold * abs(old_ema)
        ) or (self.success_ema >= self.success_threshold)
        decline_condition = current_diff < -self.reward_improve_threshold * abs(old_ema)

        # 정체 여부 판단
        if abs(current_diff) < self.stable_tolerance:
            self.stable_count += 1
        else:
            self.stable_count = 0

        # epsilon 업데이트 로직
        if improvement_condition:
            # 성능 개선
            new_epsilon = self.epsilon * self.decrease_factor
            self.epsilon = max(self.eps_min, new_epsilon)
        elif decline_condition:
            # 성능 악화
            new_epsilon = self.epsilon * self.increase_factor
            self.epsilon = min(self.eps_max, new_epsilon)
        else:
            # 개선도 악화도 없는 정체 구간
            if self.stable_count >= self.patience:
                # 정체가 오래 지속되면 탐색률을 약간 조정
                # 여기서는 임시로 epsilon을 소폭 감소시켜서 새로운 시도 유도
                new_epsilon = self.epsilon * 0.95
                self.epsilon = max(self.eps_min, new_epsilon)
                self.stable_count = 0  # stable_count 초기화
