import math
import numpy as np
from support_motion import rotation_matrix
IMU_CONTEXT_DIM = 25

class CausalMotionContext:

    def __init__(self, ap_ids):
        self.ap_ids = list(ap_ids)
        self.lookup = {a: i for i, a in enumerate(ap_ids)}
        self.first_rotation = None
        self.rotation = np.eye(3)
        self.last_imu_ns = None
        self.first_ns = None
        self.previous_wifi_ns = None
        self.previous_wifi_rotation = None
        self.acc = []
        self.gyro = []
        self.tokens = []
        self.observation_ids = []
        self.previous_rssi = np.zeros(len(ap_ids))
        self.previous_delta = np.zeros(len(ap_ids))
        self.previous_mask = np.zeros(len(ap_ids), bool)
        self.previous_ap_ns = np.zeros(len(ap_ids), np.int64)
        self.latest_imu = np.zeros(6)

    def consume(self, event):
        ns = event.monotonic_timestamp_ns
        if self.first_ns is None:
            self.first_ns = ns
        if event.kind == 'imu':
            p = event.payload
            R = rotation_matrix(p['game_rotation_quat'])
            if self.first_rotation is None:
                self.first_rotation = R.copy()
            self.rotation = R
            self.last_imu_ns = ns
            relative = self.first_rotation.T @ R
            acceleration = np.asarray(p['acc_xyz']) - 9.80665 * R[2, :]
            self.latest_imu = np.r_[relative @ acceleration, relative @ np.asarray(p['gyro_xyz'])]
            self.acc.append(self.latest_imu[:3])
            self.gyro.append(self.latest_imu[3:])
            return None
        if event.kind != 'wifi':
            return None
        relative = np.eye(3) if self.first_rotation is None else self.first_rotation.T @ self.rotation
        delta = np.eye(3) if self.previous_wifi_rotation is None else self.previous_wifi_rotation.T @ self.rotation
        angle = math.atan2(delta[1, 0], delta[0, 0])
        dt = 0.0 if self.previous_wifi_ns is None else (ns - self.previous_wifi_ns) / 1000000000.0
        age = 30.0 if self.last_imu_ns is None else max(0.0, (ns - self.last_imu_ns) / 1000000000.0)
        acceleration = np.array(self.acc) if self.acc else self.latest_imu[None, :3]
        gyro = np.array(self.gyro) if self.gyro else self.latest_imu[None, 3:]
        imu = np.r_[relative[:, :2].reshape(-1), math.sin(angle), math.cos(angle), self.rotation[2, :], np.clip(acceleration.mean(0) / 5, -4, 4), np.clip(acceleration.std(0) / 5, 0, 4), np.clip(gyro.mean(0) / 3, -4, 4), min(dt / 5, 6), min(age / 5, 6), float(self.last_imu_ns is not None), min(len(self.acc) / 100, 5), min((ns - self.first_ns) / 1000000000.0 / 60, 5)]
        assert len(imu) == IMU_CONTEXT_DIM
        ap_age = np.where(self.previous_mask, np.minimum((ns - self.previous_ap_ns) / 1000000000.0 / 10, 6), 6)
        history = np.stack((self.previous_rssi / 100, self.previous_mask.astype(float), self.previous_delta / 20, ap_age), -1).reshape(-1)
        token = np.r_[imu, history].astype(np.float32)
        if not np.isfinite(token).all():
            raise FloatingPointError('Nonfinite causal motion token')
        self.tokens.append(token)
        self.observation_ids.append(event.observation_id)
        for item in event.payload['ap_observations']:
            index = self.lookup.get(item['ap_id_hash'])
            if index is None:
                continue
            value = float(item['rssi_dbm'])
            self.previous_delta[index] = value - self.previous_rssi[index] if self.previous_mask[index] else 0.0
            self.previous_rssi[index] = value
            self.previous_mask[index] = True
            self.previous_ap_ns[index] = ns
        self.previous_wifi_ns = ns
        self.previous_wifi_rotation = None if self.first_rotation is None else self.rotation.copy()
        self.acc = []
        self.gyro = []
        return token

def trajectory_motion_tokens(events, ap_ids):
    tracker = CausalMotionContext(ap_ids)
    for event in events:
        tracker.consume(event)
    return np.asarray(tracker.tokens, np.float32)
