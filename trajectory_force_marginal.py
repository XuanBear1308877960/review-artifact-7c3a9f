import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.special import i0e, i1e
from engine_utils import write
from trajectory_axial_motion import feature_speeds
from trajectory_dual_cs import DualAxialPath

def force_context(evidence, query):
    token = np.asarray(evidence['motion'], float)
    t = np.asarray(evidence['times'], float)
    columns = token[:, :6].reshape(-1, 3, 2)
    R = np.concatenate((columns, np.cross(columns[..., 0], columns[..., 1])[:, :, None]), -1)
    valid = (token[:, 22] > 0.5) & (token[:, 21] * 5 <= 0.5)
    if not valid.any():
        return (np.zeros((len(query), 2)), np.zeros(len(query), bool))
    gravity = np.einsum('nij,nj->ni', R[valid], token[valid, 8:11]).mean(0)
    gravity /= max(float(np.linalg.norm(gravity)), 1e-08)
    ex = np.array([1.0, 0, 0])
    ex -= gravity * (ex @ gravity)
    if np.linalg.norm(ex) < 0.2:
        ex = np.array([0.0, 1, 0])
        ex -= gravity * (ex @ gravity)
    ex /= max(float(np.linalg.norm(ex)), 1e-08)
    ey = np.cross(gravity, ex)
    acc = token[:, 11:14] * 5
    acc = np.stack((acc @ ex, acc @ ey), -1)
    dt = np.diff(np.r_[t[0] - 0.1, t]).clip(0.025, 0.5)
    weight = np.exp(-0.5 * ((query[:, None] - t[None]) / 0.35) ** 2) * valid[None] * dt[None]
    weight /= weight.sum(-1, keepdims=True).clip(1e-08)
    available = (query >= t[valid][0] + 0.7) & (query <= t[valid][-1] - 0.7)
    return (weight @ acc, available)

def force_energy(A, Y, weight, parameters, derivative=False, observation_model=False):
    gain, sigma, bias = np.exp(parameters)
    sigma2 = sigma * sigma
    mass = float(weight.sum())
    W = np.diag(weight) - np.outer(weight, weight) / (mass + sigma2 / (bias * bias))
    WA = W @ A
    WY = W @ Y
    C = A.T @ WY
    dot = float(C[0, 0] + C[1, 1])
    cross = float(C[0, 1] - C[1, 0])
    strength = math.hypot(dot, cross)
    concentration = gain * strength / sigma2
    logBessel = math.log(float(i0e(concentration))) + concentration
    square = (A * WA).sum() + gain * gain * (Y * WY).sum() if observation_model else (Y * WY).sum() + gain * gain * (A * WA).sum()
    energy = 0.5 * float(square) / sigma2 - logBessel
    energy += len(A) * math.log(2 * math.pi) + len(A) * math.log(sigma2) - np.log(weight).sum() + math.log1p(bias * bias * mass / sigma2)
    if not derivative:
        return float(energy)
    angle = math.atan2(cross, dot)
    c, s = (math.cos(angle), math.sin(angle))
    rotation = np.array([[c, s], [-s, c]])
    resultant = float(i1e(concentration) / i0e(concentration))
    gradient = ((gain * gain if observation_model else 1.0) * WY - gain * resultant * WA @ rotation) / sigma2
    return (float(energy), gradient, dict(force_rotation_concentration=concentration, force_rotation_resultant=resultant, force_frame_rotation_mode_degrees=math.degrees(angle), motion_heading_not_observed=True, force_gain=gain, force_noise_sigma=sigma, force_bias_sigma=bias, observation_model=observation_model))

def source_rows(a):
    source = json.loads((a.capture / 'segments.json').read_text())['source_train']
    evidence = feature_speeds(a.features)
    labels = {}
    rows = []
    for ordinal, s in enumerate(source):
        name = s['device']
        if name not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target label in force training')
        if name not in labels:
            labels[name] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / name / 'evaluation_labels.json').read_text())
        folder = a.capture / 'source_train' / s['trajectory_id']
        pred = json.loads((folder / 'predictions.json').read_text())
        selected = [r for r in pred if r['observation_id'] in labels[name]]
        t = np.r_[0.0, [(r['timestamp_ns'] - s['start_ns']) / 1000000000.0 for r in selected]]
        xy = np.vstack((s['initial_position_xy_m'], [labels[name][r['observation_id']]['gt_xy_m'] for r in selected]))
        with np.load(folder / 'CS_training_contexts.npz', allow_pickle=False) as z:
            cs_end = float((z['timestamp_ns'][-1] - s['start_ns']) / 1000000000.0)
        end = max(float(t[-1]), cs_end)
        knots = np.linspace(0, end, int(np.ceil(end / 0.5)) + 1)
        query = 0.5 * (knots[:-1] + knots[1:])
        force, valid = force_context(evidence['source_train', s['trajectory_id']], query)
        grid = np.arange(t[0], t[-1] + 0.001, 0.1)
        location = np.stack([np.interp(grid, t, xy[:, j]) for j in range(2)], -1)
        smooth = gaussian_filter1d(location, 3.5, axis=0, mode='nearest')
        acc = CubicSpline(grid, smooth, axis=0)(query, 2)
        valid &= (query >= 0.7) & (query <= t[-1] - 0.7)
        if valid.sum() < 5:
            continue
        rows.append(dict(force=force[valid], acceleration=acc[valid], weight=np.diff(knots)[valid], heldout=ordinal % 5 == 0))
    return rows

def fit(a):
    rows = source_rows(a)

    def objective(p, heldout):
        total = 0.0
        count = 0
        for r in rows:
            if r['heldout'] != heldout:
                continue
            total += force_energy(r['force'], r['acceleration'], r['weight'], p, observation_model=a.observation_model)
            count += 2 * len(r['force'])
        return total / count
    result = minimize(objective, np.log([0.5, 0.6, 0.1]), args=(False,), method='L-BFGS-B', bounds=[(math.log(0.001), math.log(3.0)), (math.log(0.05), math.log(8.0)), (math.log(0.01), math.log(4.0))])

    def baseline(p, heldout):
        return objective(np.r_[math.log(1e-12), p], heldout)
    zero = minimize(baseline, np.log([0.6, 0.1]), args=(False,), method='L-BFGS-B', bounds=[(math.log(0.05), math.log(8.0)), (math.log(0.01), math.log(4.0))])
    measured = objective(result.x, True)
    reference = baseline(zero.x, True)
    prior = json.loads(a.prior.read_text())
    prior.update(marginal_force_parameters=result.x.tolist(), marginal_force_heldout_NLL=measured, marginal_force_zero_baseline_heldout_NLL=reference, marginal_force_source_only=True, marginal_force_unknown_global_rotation_integrated=True, phone_yaw_is_motion_heading=False, marginal_force_replaces_local_neural_acceleration=True, marginal_force_observation_model=a.observation_model, marginal_force_selected=bool(measured < reference))
    write(a.output, prior)
    print(json.dumps(dict(source_heldout_NLL=measured, zero_force_heldout_NLL=reference, selected=prior['marginal_force_selected'], gain=float(np.exp(result.x[0])), force_noise_sigma=float(np.exp(result.x[1])), force_bias_sigma=float(np.exp(result.x[2])))), flush=True)

class ForceDualPath(DualAxialPath):

    def __init__(self, *args, **kwargs):
        evidence = kwargs.pop('force_context')
        super().__init__(*args, **kwargs)
        A, valid = force_context(evidence, self.mid)
        self.force_ix = np.where(valid)[0]
        self.measured_force = A[valid]
        self.force_matrix = self.A[self.force_ix]
        self.force_weight = self.adt[self.force_ix]
        self.force_intervals = None
        if 'force_preintegration_minimum_span_s' in self.prior:
            from trajectory_force_preintegration import measured_intervals
            intervals, self.measured_force = measured_intervals(evidence, self.prior['force_preintegration_minimum_span_s'])
            width = intervals[:, 1] - intervals[:, 0]
            self.force_matrix = (self.spline(intervals[:, 1], 1) - self.spline(intervals[:, 0], 1)) / width[:, None]
            self.force_weight = width
            self.force_intervals = intervals
            covered = ((self.mid[:, None] >= intervals[None, :, 0]) & (self.mid[:, None] <= intervals[None, :, 1])).any(-1)
            self.force_ix = np.where(covered)[0]
        self.force_parameters = np.asarray(self.prior['marginal_force_parameters'])
        if not self.prior['marginal_force_selected']:
            raise ValueError('Force evidence failed source selection')

    def path_objective(self, vector, mode):
        value, gradient = super().path_objective(vector, mode)
        if len(self.measured_force) < 5:
            return (value, gradient)
        q, _ = self.unpack(vector)
        ix = self.force_ix
        A = self.A[ix]
        weight = self.adt[ix]
        acceleration = A @ q
        old = acceleration.copy()
        scale2 = self.prior['acceleration_scale_m_s2'] ** 2
        if self.prior.get('learned_acceleration_center', False):
            old -= self.learned_acceleration[ix]
            scale2 = self.prior['learned_acceleration_sigma_m_s2'] ** 2
        square = np.square(old).sum(-1)
        balance = self.prior.get('trajectory_factor_weights', {})
        old_weight = balance.get('acceleration', 1.0)
        force_weight = balance.get('force', 1.0)
        value -= float((2.5 * np.log1p(square / (3 * scale2)) * weight).sum()) * old_weight
        g_old = A.T @ (5 * old / (3 * scale2 + square)[:, None] * weight[:, None]) * old_weight
        loss, g_force, _ = force_energy(self.measured_force, self.force_matrix @ q, self.force_weight, self.force_parameters, True, self.prior.get('marginal_force_observation_model', False))
        value += loss * force_weight
        grad = self.force_matrix.T @ g_force * force_weight - g_old
        gradient[:2 * (self.n - 1)] += grad[1:].ravel()
        return (value, gradient)

    def solve_axis(self, alternative, modes):
        predictions, diagnostic = super().solve_axis(alternative, modes)
        if len(self.measured_force) >= 5:
            q, _ = self.unpack(self.last_selected_vector[:-1])
            ix = self.force_ix
            _, _, d = force_energy(self.measured_force, self.force_matrix @ q, self.force_weight, self.force_parameters, True, self.prior.get('marginal_force_observation_model', False))
            diagnostic['force_evidence'] = d
            if self.force_intervals is not None:
                diagnostic['force_evidence'].update(nonoverlapping_actual_interval_count=len(self.force_intervals), actual_interval_bounds_s=self.force_intervals.tolist(), known_phone_heading_used=False)
        return (predictions, diagnostic)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'features', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--observation-model', action='store_true')
    a = p.parse_args()
    fit(a)
if __name__ == '__main__':
    main()
