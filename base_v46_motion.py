import numpy as np

def rotation_matrix(q):
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    if n < 1e-08:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

class MotionPredictor:

    def __init__(self):
        self.accel = np.zeros(2)
        self.gyro = 0.0
        self.game_yaw = 0.0
        self.available = False
        self.reference_yaw = 0.0
        self.reference_elapsed_s = 0.0
        self.reference_steps = 0

    def observe(self, payload):
        R = rotation_matrix(payload['game_rotation_quat'])
        self.accel = np.clip((R @ np.array(payload['acc_xyz']))[:2], -5, 5)
        self.game_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        self.gyro = float((R @ np.array(payload['gyro_xyz']))[2])
        self.available = True
        self.reference_yaw = self.game_yaw
        self.reference_elapsed_s = 0.0

    def advance_reference(self, dt, cfg, disabled=False):
        if cfg.get('gyro_consistent_game_frame', False) and self.available and (not disabled) and (dt > 0):
            self.reference_yaw += self.gyro * dt
            self.reference_elapsed_s += dt
            self.reference_steps += 1

    def predict(self, state, cov, dt, cfg, disabled=False):
        d = len(state)
        F = np.eye(d)
        rate = cfg.get('velocity_relaxation_rate_hz', 0.0)
        decay = np.exp(-rate * dt) if rate > 1e-06 else 1.0
        velocity_dt = -np.expm1(-rate * dt) / rate if rate > 1e-06 else dt
        acceleration_dt = (dt - velocity_dt) / rate if rate > 1e-06 else 0.5 * dt ** 2
        F[0, 2] = F[1, 3] = velocity_dt
        F[2, 2] = F[3, 3] = decay
        x = F @ state
        if self.available and (not disabled):
            reference = self.reference_yaw if cfg.get('gyro_consistent_game_frame', False) else self.game_yaw
            angle = state[4] - reference
            c, s = (np.cos(angle), np.sin(angle))
            rotation = np.array([[c, -s], [s, c]])
            acceleration = rotation @ self.accel
            derivative = np.array([[-s, -c], [c, -s]]) @ self.accel
            x[:2] += acceleration * acceleration_dt
            x[2:4] += acceleration * velocity_dt
            x[4] += self.gyro * dt
            F[:2, 4] = derivative * acceleration_dt
            F[2:4, 4] = derivative * velocity_dt
        Q = np.zeros((d, d))
        qa = cfg['acceleration_process_noise'] ** 2
        if cfg.get('acceleration_noise_model') == 'legacy_per_event':
            B = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
            Q[:4, :4] = qa * (B @ B.T)
        elif rate > 1e-06:
            qvv = -np.expm1(-2 * rate * dt) / (2 * rate)
            qpv = velocity_dt ** 2 / 2
            qpp = (dt - 2 * velocity_dt + qvv) / rate ** 2
            Q[:2, :2] = np.eye(2) * qa * max(0.0, qpp)
            Q[:2, 2:4] = Q[2:4, :2] = np.eye(2) * qa * qpv
            Q[2:4, 2:4] = np.eye(2) * qa * qvv
        else:
            Q[:2, :2] = np.eye(2) * qa * dt ** 3 / 3
            Q[:2, 2:4] = Q[2:4, :2] = np.eye(2) * qa * dt ** 2 / 2
            Q[2:4, 2:4] = np.eye(2) * qa * dt
        Q[4, 4] = 0.05 ** 2 * dt
        Q[5:9, 5:9] = np.eye(4) * cfg['bias_process_noise'] * dt
        temporal = cfg.get('cs_temporal_noise')
        latent_end = d - 4 if temporal else d
        Q[9:latent_end, 9:latent_end] = np.eye(latent_end - 9) * cfg['latent_process_noise'] * dt
        if temporal:
            for j, parameters in enumerate(temporal.values()):
                index = latent_end + j
                decay_noise = np.exp(-dt / parameters['tau_s'])
                F[index, index] = decay_noise
                x[index] *= decay_noise
                Q[index, index] = parameters['stationary_sigma_m'] ** 2 * -np.expm1(-2 * dt / parameters['tau_s'])
        return (x, F @ cov @ F.T + Q)

    def predict_batch(self, state, cov, dt, cfg, disabled=False):
        state = np.asarray(state, dtype=float)
        d = state.shape[-1]
        F = np.broadcast_to(np.eye(d), (*state.shape[:-1], d, d)).copy()
        rate = cfg.get('velocity_relaxation_rate_hz', 0.0)
        decay = np.exp(-rate * dt) if rate > 1e-06 else 1.0
        velocity_dt = -np.expm1(-rate * dt) / rate if rate > 1e-06 else dt
        acceleration_dt = (dt - velocity_dt) / rate if rate > 1e-06 else 0.5 * dt ** 2
        F[..., 0, 2] = F[..., 1, 3] = velocity_dt
        F[..., 2, 2] = F[..., 3, 3] = decay
        x = (F @ state[..., None]).squeeze(-1)
        if self.available and (not disabled):
            reference = self.reference_yaw if cfg.get('gyro_consistent_game_frame', False) else self.game_yaw
            angle = state[..., 4] - reference
            c, s = (np.cos(angle), np.sin(angle))
            ax, ay = self.accel
            acceleration = np.stack((c * ax - s * ay, s * ax + c * ay), -1)
            derivative = np.stack((-s * ax - c * ay, c * ax - s * ay), -1)
            x[..., :2] += acceleration * acceleration_dt
            x[..., 2:4] += acceleration * velocity_dt
            x[..., 4] += self.gyro * dt
            F[..., :2, 4] = derivative * acceleration_dt
            F[..., 2:4, 4] = derivative * velocity_dt
        Q = np.zeros((d, d))
        qa = cfg['acceleration_process_noise'] ** 2
        if cfg.get('acceleration_noise_model') == 'legacy_per_event':
            B = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
            Q[:4, :4] = qa * (B @ B.T)
        elif rate > 1e-06:
            qvv = -np.expm1(-2 * rate * dt) / (2 * rate)
            qpv = velocity_dt ** 2 / 2
            qpp = (dt - 2 * velocity_dt + qvv) / rate ** 2
            Q[:2, :2] = np.eye(2) * qa * max(0.0, qpp)
            Q[:2, 2:4] = Q[2:4, :2] = np.eye(2) * qa * qpv
            Q[2:4, 2:4] = np.eye(2) * qa * qvv
        else:
            Q[:2, :2] = np.eye(2) * qa * dt ** 3 / 3
            Q[:2, 2:4] = Q[2:4, :2] = np.eye(2) * qa * dt ** 2 / 2
            Q[2:4, 2:4] = np.eye(2) * qa * dt
        Q[4, 4] = 0.05 ** 2 * dt
        Q[5:9, 5:9] = np.eye(4) * cfg['bias_process_noise'] * dt
        temporal = cfg.get('cs_temporal_noise')
        latent_end = d - 4 if temporal else d
        Q[9:latent_end, 9:latent_end] = np.eye(latent_end - 9) * cfg['latent_process_noise'] * dt
        if temporal:
            for j, parameters in enumerate(temporal.values()):
                index = latent_end + j
                decay_noise = np.exp(-dt / parameters['tau_s'])
                F[..., index, index] = decay_noise
                x[..., index] *= decay_noise
                Q[index, index] = parameters['stationary_sigma_m'] ** 2 * -np.expm1(-2 * dt / parameters['tau_s'])
        return (x, F @ cov @ F.swapaxes(-1, -2) + Q)
