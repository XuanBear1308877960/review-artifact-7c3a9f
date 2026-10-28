from collections import deque
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from base_model import mlp
from base_v46_motion import MotionPredictor
KIND = 'causal_IMU_magnitude_speed'

class SpeedContext:

    def __init__(self):
        self.rows = deque()

    def observe(self, event):
        if event['kind'] != 'imu':
            return
        now = event['monotonic_timestamp_ns'] / 1000000000.0
        payload = event['payload']
        acceleration = float(np.linalg.norm(payload['acc_xyz'])) - 9.80665
        turn_rate = float(np.linalg.norm(payload['gyro_xyz']))
        self.rows.append((now, np.clip(acceleration, -15, 15), np.clip(turn_rate, 0, 12)))
        while self.rows and now - self.rows[0][0] > 2.5:
            self.rows.popleft()

    def features(self, now_ns):
        now = now_ns / 1000000000.0
        rows = np.asarray(self.rows, float).reshape(-1, 3)
        result = []
        for window in (0.4, 1.0, 2.0):
            use = rows[(rows[:, 0] <= now) & (rows[:, 0] >= now - window)]
            if not len(use):
                result.extend([0.0] * 9)
                continue
            a, w = (use[:, 1], use[:, 2])
            result.extend([a.mean() / 5, a.std() / 5, np.sqrt(np.mean(a * a)) / 5, np.quantile(abs(a), 0.95) / 10, np.sqrt(np.mean(w * w)) / 3, np.quantile(w, 0.95) / 3, (use[-1, 0] - use[0, 0]) / window, min(len(use) / (20 * window), 2.0), 1.0])
        return np.asarray(result, np.float32)

class SpeedHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = mlp(27, 32, 16, 2)

    def forward(self, features):
        value = self.network(features)
        speed = F.softplus(value[..., 0]).clamp_max(4.0)
        sigma = (0.25 + F.softplus(value[..., 1])).clamp_max(2.0)
        return (speed, sigma)

class SpeedMagnitudeMotion(MotionPredictor):

    def __init__(self):
        super().__init__()
        self.target_speed = None
        self.speed_sigma = 1.0

    def observe(self, payload):
        self.available = False

    def predict(self, state, cov, dt, cfg, disabled=False):
        x, p = self.predict_batch(np.asarray(state)[None], np.asarray(cov)[None], dt, cfg)
        return (x[0], p[0])

    def predict_batch(self, state, cov, dt, cfg, disabled=False):
        radio_cfg = dict(cfg, velocity_relaxation_rate_hz=0.0)
        if self.target_speed is None:
            return super().predict_batch(state, cov, dt, radio_cfg, disabled=True)
        state = np.asarray(state, float)
        x = state.copy()
        d = state.shape[-1]
        Fm = np.broadcast_to(np.eye(d), (*state.shape[:-1], d, d)).copy()
        velocity = state[..., 2:4]
        norm = np.linalg.norm(velocity, axis=-1)
        unit = velocity / np.maximum(norm[..., None], 0.2)
        fallback = np.stack((np.cos(state[..., 4]), np.sin(state[..., 4])), -1)
        unit = np.where((norm > 0.2)[..., None], unit, fallback)
        derivative = (np.eye(2) - unit[..., :, None] * unit[..., None, :]) / np.maximum(norm[..., None, None], 0.2)
        derivative = np.where((norm > 0.2)[..., None, None], derivative, 0.0)
        rate = 1.0 / (1.0 + self.speed_sigma ** 2)
        decay = math.exp(-rate * dt)
        velocity_dt = -math.expm1(-rate * dt) / rate
        pos_control = dt - velocity_dt
        speed = self.target_speed
        x[..., :2] += velocity_dt * velocity + pos_control * speed * unit
        x[..., 2:4] = decay * velocity + (1 - decay) * speed * unit
        Fm[..., :2, 2:4] = velocity_dt * np.eye(2) + pos_control * speed * derivative
        Fm[..., 2:4, 2:4] = decay * np.eye(2) + (1 - decay) * speed * derivative
        turn = np.stack((-unit[..., 1], unit[..., 0]), -1)
        Fm[..., :2, 4] = np.where((norm <= 0.2)[..., None], pos_control * speed * turn, 0.0)
        Fm[..., 2:4, 4] = np.where((norm <= 0.2)[..., None], (1 - decay) * speed * turn, 0.0)
        _, process = super().predict_batch(state, np.zeros_like(cov), dt, radio_cfg, disabled=True)
        return (x, Fm @ cov @ Fm.swapaxes(-1, -2) + process)
