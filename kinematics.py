import math
import numpy as np
import torch
from base_native_bias_state import NativeEngine
from base_v46_motion import MotionPredictor
from base_v46_motion import rotation_matrix
from speed_magnitude import SpeedContext, SpeedMagnitudeMotion

class RadioMotionPredictor(MotionPredictor):

    def observe(self, payload):
        self.available = False

    def predict(self, state, covariance, dt, cfg, disabled=False):
        return super().predict(state, covariance, dt, cfg, disabled=True)

    def predict_batch(self, state, covariance, dt, cfg, disabled=False):
        return super().predict_batch(state, covariance, dt, cfg, disabled=True)

class GameFrameMotion(MotionPredictor):

    def observe(self, payload):
        rotation = rotation_matrix(payload['game_rotation_quat'])
        self.accel = np.clip((rotation @ np.asarray(payload['acc_xyz']))[:2], -5, 5)
        self.gyro = self.game_yaw = self.reference_yaw = 0.0
        self.available = True

class FrameOffsetMotion(MotionPredictor):

    def __init__(self):
        super().__init__()
        self.initial_game_yaw = None

    def observe(self, payload):
        rotation = rotation_matrix(payload['game_rotation_quat'])
        if self.initial_game_yaw is None:
            self.initial_game_yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        acceleration = (rotation @ np.asarray(payload['acc_xyz']))[:2]
        c, s = (np.cos(self.initial_game_yaw), np.sin(self.initial_game_yaw))
        self.accel = np.clip([c * acceleration[0] + s * acceleration[1], -s * acceleration[0] + c * acceleration[1]], -5, 5)
        self.gyro = self.game_yaw = self.reference_yaw = 0.0
        self.available = True

class BiasFrameMotion(FrameOffsetMotion):

    def predict(self, state, covariance, dt, cfg, disabled=False):
        mean, cov = self.predict_batch(np.asarray(state)[None], np.asarray(covariance)[None], dt, cfg, disabled)
        return (mean[0], cov[0])

    def predict_batch(self, state, covariance, dt, cfg, disabled=False):
        x = np.asarray(state, float).copy()
        d = x.shape[-1]
        F = np.broadcast_to(np.eye(d), (*x.shape[:-1], d, d)).copy()
        F[..., 0, 2] = F[..., 1, 3] = dt
        x[..., :2] += x[..., 2:4] * dt
        if self.available:
            angle = state[..., 4]
            c, s = (np.cos(angle), np.sin(angle))
            rotation = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
            acc = np.einsum('...ij,...j->...i', rotation, self.accel - state[..., 15:17])
            derivative = np.stack([-acc[..., 1], acc[..., 0]], -1)
            x[..., :2] += acc * (0.5 * dt * dt)
            x[..., 2:4] += acc * dt
            F[..., :2, 4] = derivative * (0.5 * dt * dt)
            F[..., 2:4, 4] = derivative * dt
            F[..., :2, 15:17] = -rotation * (0.5 * dt * dt)
            F[..., 2:4, 15:17] = -rotation * dt
        else:
            rotation = np.broadcast_to(np.eye(2), (*x.shape[:-1], 2, 2))
        Q = np.zeros_like(F)
        eye = np.eye(2)
        qa, qb = (cfg['acceleration_process_noise'] ** 2, 0.02 ** 2)
        Q[..., :2, :2] = eye * (qa * dt ** 3 / 3 + qb * dt ** 5 / 20)
        Q[..., :2, 2:4] = Q[..., 2:4, :2] = eye * (qa * dt ** 2 / 2 + qb * dt ** 4 / 8)
        Q[..., 2:4, 2:4] = eye * (qa * dt + qb * dt ** 3 / 3)
        Q[..., :2, 15:17] = -rotation * (qb * dt ** 3 / 6)
        Q[..., 15:17, :2] = Q[..., :2, 15:17].swapaxes(-1, -2)
        Q[..., 2:4, 15:17] = -rotation * (qb * dt ** 2 / 2)
        Q[..., 15:17, 2:4] = Q[..., 2:4, 15:17].swapaxes(-1, -2)
        Q[..., 15:17, 15:17] = eye * qb * dt
        Q[..., 4, 4] = 0.05 ** 2 * dt
        Q[..., 5:9, 5:9] = np.eye(4) * cfg['bias_process_noise'] * dt
        Q[..., 9:15, 9:15] = np.eye(6) * cfg['latent_process_noise'] * dt
        return (x, F @ covariance @ F.swapaxes(-1, -2) + Q)

class GravityPlaneMotion(MotionPredictor):

    def observe(self, payload):
        if np.linalg.norm(payload['game_rotation_quat']) < 0.1:
            self.available = False
            return
        gravity = rotation_matrix(payload['game_rotation_quat'])[2]
        tilt = np.asarray([1 + gravity[2], gravity[1], -gravity[0], 0.0])
        if np.linalg.norm(tilt) < 1e-06:
            tilt = np.asarray([0.0, 1.0, 0.0, 0.0])
        self.accel = np.clip((rotation_matrix(tilt) @ np.asarray(payload['acc_xyz']))[:2], -5, 5)
        self.gyro = self.game_yaw = self.reference_yaw = 0.0
        self.available = True

    def predict_batch(self, state, covariance, dt, cfg, disabled=False):
        mean, cov = super().predict_batch(state, covariance, dt, cfg, disabled)
        cov[..., 4, 4] += (0.35 ** 2 - 0.05 ** 2) * dt
        return (mean, cov)

class ControlMixtureMotion(MotionPredictor):

    def predict_batch(self, state, covariance, dt, cfg, disabled=False):
        split = state.shape[0] // 2
        imu_x, imu_p = super().predict_batch(state[:split], covariance[:split], dt, cfg, disabled)
        radio_cfg = dict(cfg, velocity_relaxation_rate_hz=0.0)
        radio_x, radio_p = super().predict_batch(state[split:], covariance[split:], dt, radio_cfg, disabled=True)
        return (np.concatenate([imu_x, radio_x]), np.concatenate([imu_p, radio_p]))

class RadioDirectionEngine(NativeEngine):
    last_instance = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.net.orientation_basis:
            raise ValueError("This experiment preserves epoch291's heading-independent radio field")
        self.direction_policy = self.filter_cfg['direction_policy']
        self.speed_controlled = self.direction_policy in ('radio_speed_magnitude', 'radio_speed_centered')
        if self.direction_policy in ('radio_velocity_mixture', 'radio_constant_velocity', 'radio_temporal_velocity_mixture', 'radio_temporal_constant_velocity', 'radio_speed_magnitude', 'radio_velocity_centered', 'radio_speed_centered'):
            self.motion = RadioMotionPredictor()
            self.filter_cfg = dict(self.filter_cfg, velocity_relaxation_rate_hz=0.0)
        if self.direction_policy in ('radio_velocity_mixture', 'radio_temporal_velocity_mixture', 'radio_speed_magnitude', 'radio_velocity_centered', 'radio_speed_centered'):
            within_fraction = 0.25
            angle = torch.arange(self.k, device=self.device, dtype=self.mean.dtype) * (2 * math.pi / self.k)
            radius = math.sqrt(2 * (1 - within_fraction))
            self.mean[:, 2] = radius * angle.cos()
            self.mean[:, 3] = radius * angle.sin()
            self.cov[:, 2:4, 2:4] = torch.eye(2, device=self.device) * within_fraction
        elif self.direction_policy in ('radio_frame_registration', 'radio_frame_registration_aligned', 'radio_spatial_basin', 'radio_balanced', 'radio_balanced_median', 'radio_gravity_plane', 'radio_gravity_plane_centered', 'radio_frame16', 'radio_frame32', 'radio_frame_median', 'radio_frame_risk_readout', 'radio_frame_independent_cs'):
            self.motion = FrameOffsetMotion()
            if self.direction_policy in ('radio_frame_registration', 'radio_frame_median', 'radio_frame_risk_readout', 'radio_frame_independent_cs'):
                self.motion = GameFrameMotion()
            if self.direction_policy == 'radio_frame_median':
                self.filter_cfg = dict(self.filter_cfg, spatial_point_rule='geometric_median')
            if self.direction_policy == 'radio_frame_risk_readout':
                self.filter_cfg = dict(self.filter_cfg, position_readout='posterior_Euclidean_risk')
            if self.direction_policy in ('radio_gravity_plane', 'radio_gravity_plane_centered'):
                self.motion = GravityPlaneMotion()
            if self.direction_policy == 'radio_spatial_basin':
                self.filter_cfg = dict(self.filter_cfg, spatial_basin_policy='mass_mode')
            if self.direction_policy == 'radio_balanced_median':
                self.filter_cfg = dict(self.filter_cfg, spatial_point_rule='geometric_median')
        elif self.direction_policy == 'radio_spatial_basin_cv':
            self.motion = RadioMotionPredictor()
            self.filter_cfg = dict(self.filter_cfg, velocity_relaxation_rate_hz=0.0, spatial_basin_policy='mass_mode')
        elif self.direction_policy == 'radio_frame_bias_cv':
            if self.temporal:
                raise ValueError('Bias-state layout is defined for the selected non-OU epoch291')
            self.motion = BiasFrameMotion()
            self.dim = 17
            self.mean = torch.nn.functional.pad(self.mean, (0, 2))
            self.cov = torch.nn.functional.pad(self.cov, (0, 2, 0, 2))
            self.cov[:, 15:17, 15:17] = torch.eye(2, device=self.device) * 0.35 ** 2
            self.filter_cfg = dict(self.filter_cfg, velocity_relaxation_rate_hz=0.0)
        elif self.direction_policy == 'radio_model_mixture':
            old_k = self.k

            def double_modes(value):
                if torch.is_tensor(value) and value.ndim and (value.shape[0] == old_k):
                    return torch.cat([value, value], dim=0)
                if isinstance(value, dict):
                    return {k: double_modes(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [double_modes(v) for v in value]
                return value
            for key, value in list(vars(self).items()):
                if torch.is_tensor(value) or isinstance(value, (dict, list)):
                    setattr(self, key, double_modes(value))
            self.k = old_k * 2
            self.log_weight[:old_k] = math.log(0.25 / old_k)
            self.log_weight[old_k:] = math.log(0.75 / old_k)
            self.motion = ControlMixtureMotion()
        elif self.direction_policy == 'control_independent_cs':
            pass
        elif self.direction_policy not in ('radio_constant_velocity', 'radio_temporal_constant_velocity'):
            raise ValueError('Unknown direction policy')
        self.radio_position_history = []
        self.radio_contexts = []
        if self.direction_policy in ('radio_frame_independent_cs', 'control_independent_cs'):
            self.filter_cfg = dict(self.filter_cfg, cs_radio_feedback_policy='state_only')
        if self.direction_policy in ('radio_frame16', 'radio_frame32'):
            new_k = int(self.direction_policy[len('radio_frame'):])
            old_k = self.k
            index = torch.arange(new_k, device=self.device) % old_k

            def resize_modes(value):
                if torch.is_tensor(value) and value.ndim and (value.shape[0] == old_k):
                    return value.index_select(0, index)
                if isinstance(value, dict):
                    return {key: resize_modes(v) for key, v in value.items()}
                if isinstance(value, list):
                    return [resize_modes(v) for v in value]
                return value
            for key, value in list(vars(self).items()):
                if torch.is_tensor(value) or isinstance(value, (dict, list)):
                    setattr(self, key, resize_modes(value))
            self.k = new_k
            self.heading = torch.arange(new_k, device=self.device) * (2 * math.pi / new_k)
            self.mean[:, 4] = self.heading
            self.cov[:, 4, 4] = (math.pi / new_k) ** 2 / 3
            self.log_weight.fill_(-math.log(new_k))
        if self.direction_policy in ('radio_velocity_centered', 'radio_speed_centered', 'radio_gravity_plane_centered'):
            self.filter_cfg = dict(self.filter_cfg, center_weak_prior_moment=True)
        if self.speed_controlled:
            self.motion = SpeedMagnitudeMotion()
            self.speed_context = SpeedContext()
            self.motion_speed_time = None
            if not hasattr(self.net, 'motion_speed_head'):
                raise ValueError('Speed control needs a source-trained magnitude model')
        RadioDirectionEngine.last_instance = self

    def _advance(self, ns):
        if getattr(self, 'speed_controlled', False):
            if self.motion_speed_time is not None and (ns - self.motion_speed_time) / 1000000000.0 > 1.5:
                self.motion.target_speed = None
        return super()._advance(ns)

    def _temporal_context(self, delta):
        if not getattr(self.net, 'direction_temporal_head', None):
            return super()._temporal_context(delta)
        rows = self.radio_position_history[-16:]

        def pad(key, shape, dtype=torch.float32):
            value = torch.stack([row[key].to(dtype=dtype) for row in rows]) if rows else torch.zeros(0, *shape, device=self.device, dtype=dtype)
            return torch.cat((torch.zeros(16 - len(rows), *shape, device=self.device, dtype=dtype), value))[None]
        now = self.timestamp / 1000000000.0
        times = [-1000.0] * (16 - len(rows)) + [row['time'] - now for row in rows]
        context = dict(rssi=pad('rssi', (self.a,)), mask=pad('mask', (self.a,), torch.bool), position=pad('position', (self.k, 2)), source=pad('source', (self.k, self.a)), correction=pad('correction', (self.k, self.a)), variance=pad('variance', (self.k,)), times=torch.tensor([times], device=self.device, dtype=torch.float32))
        if rows and (not all((row['time'] < now for row in rows))):
            raise ValueError('Radio temporal history must precede the current scan')
        self.radio_contexts.append({key: value[0].cpu().numpy().copy() for key, value in context.items()})
        return context

    def _consume_wifi(self, event):
        output = super()._consume_wifi(event)
        if output is None or not getattr(self.net, 'direction_temporal_head', None):
            return output
        forecast = self.forecasts[-1]
        context = torch.tensor(forecast['motion'], device=self.device)
        field = self.net.field(self.mean[:, :2], context.expand(self.k, 32))
        correction = (field['basis'] @ self.mean[:, self.eta, None]).squeeze(-1) + torch.tensor(forecast['short_residual'], device=self.device)
        self.radio_position_history.append(dict(time=self.timestamp / 1000000000.0, rssi=torch.tensor(forecast['rssi'], device=self.device), mask=torch.tensor(forecast['mask'], device=self.device), position=self.mean[:, :2].detach().clone(), source=field['mean'].detach().clone(), correction=correction.detach().clone(), variance=self.cov[:, :2, :2].diagonal(dim1=-2, dim2=-1).sum(-1).detach().clone()))
        self.radio_position_history = self.radio_position_history[-16:]
        return output

    @torch.no_grad()
    def consume(self, event):
        output = super().consume(event)
        if self.speed_controlled and event['kind'] == 'imu':
            self.speed_context.observe(event)
            vector = self.speed_context.features(event['monotonic_timestamp_ns'])
            if vector[6] >= 0.5:
                speed, sigma = self.net.motion_speed_head(torch.tensor(vector, device=self.device))
                self.motion.target_speed = float(speed)
                self.motion.speed_sigma = float(sigma)
                self.motion_speed_time = event['monotonic_timestamp_ns']
        if output is not None:
            weights = self.log_weight.softmax(-1)
            velocity = (weights[:, None] * self.mean[:, 2:4]).sum(0)
            output.update(velocity_xy_m_s=velocity.cpu().tolist(), mode_velocity_xy_m_s=self.mean[:, 2:4].cpu().tolist(), direction_policy=self.direction_policy, phone_heading_used_as_motion_direction=False)
            if self.direction_policy == 'radio_model_mixture':
                output['IMU_control_model_probability'] = float(weights[:self.k // 2].sum())
            if self.direction_policy == 'radio_frame_bias_cv':
                output['acceleration_bias_game_xy_m_s2'] = (weights[:, None] * self.mean[:, 15:17]).sum(0).cpu().tolist()
            if self.speed_controlled:
                output.update(IMU_speed_m_s=self.motion.target_speed, IMU_speed_sigma_m_s=self.motion.speed_sigma, IMU_speed_inputs='acceleration/gyro magnitudes only; no quaternion or heading')
        return output
