import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.special import logsumexp
import torch
from engine_utils import write
from speed_magnitude import SpeedHead, SpeedContext
from base_evaluate import AVAILABLE_CPUS
_SPEED = None
_SESSIONS = {}

def speed_evidence(base, data, s, times, return_features=False):
    global _SPEED
    if _SPEED is None:
        saved = torch.load(base / 'runs/direction_v80/speed_source_v1/trained.pt', map_location='cpu', weights_only=False)
        if set(saved['source_training_devices']) != {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Unexpected speed training phones')
        _SPEED = SpeedHead().eval()
        _SPEED.load_state_dict({k[len('motion_speed_head.'):]: v for k, v in saved['state_dict'].items() if k.startswith('motion_speed_head.')})
    name = s['device']
    if name not in _SESSIONS:
        _SESSIONS[name] = [json.loads(line) for line in (data / name / 'online.ndjson').read_text().splitlines()]
    context, x, captured = (SpeedContext(), [], [])
    for event in _SESSIONS[name]:
        ns = event['monotonic_timestamp_ns']
        if not s['start_ns'] <= ns < s['end_ns']:
            continue
        context.observe(event)
        if event['kind'] == 'wifi':
            x.append(context.features(ns))
            captured.append(ns)
    lookup = {ns: i for i, ns in enumerate(captured)}
    x = np.asarray([x[lookup[int(ns)]] for ns in times], np.float32)
    with torch.no_grad():
        mean, sigma = _SPEED(torch.from_numpy(x))
    result = (mean.numpy().astype(float), np.maximum(sigma.numpy().astype(float), 0.5), x[:, 6] >= 0.5)
    return result + (x,) if return_features else result

class SensorPath:

    def __init__(self, folder, segment, prior, speed, use_speed, learned_path=None):
        self.s, self.prior, self.use_speed = (segment, prior, use_speed)
        self.predictions = json.loads((folder / 'predictions.json').read_text())
        with np.load(folder / 'trajectory_factors.npz', allow_pickle=False) as z:
            f = {k: z[k].copy() for k in z.files}
        with np.load(folder / 'CS_training_contexts.npz', allow_pickle=False) as z:
            cs = {k: z[k].copy() for k in z.files}
        self.times_ns = f['times_ns']
        self.t = (self.times_ns - segment['start_ns']) / 1000000000.0
        end = max(float(self.t[-1]), float((cs['timestamp_ns'][-1] - segment['start_ns']) / 1000000000.0))
        self.knots = np.linspace(0.0, end, int(np.ceil(end / prior['knot_interval_s'])) + 1)
        self.spline = CubicSpline(self.knots, np.eye(len(self.knots)), axis=0, bc_type='natural')
        self.B = self.spline(self.t)
        self.V = self.spline(self.t, 1)
        mid = 0.5 * (self.knots[:-1] + self.knots[1:])
        self.A = self.spline(mid, 2)
        self.adt = np.diff(self.knots)
        self.g = f['grid_xy'].astype(float)
        self.bandwidth2 = float(prior.get('radio_kernel_bandwidth_m', 0.35)) ** 2
        scores = f['radio_loglike'].astype(float)
        self.scores = scores - scores.max(axis=(1, 2), keepdims=True)
        self.wifi_weight = np.clip(np.diff(np.r_[0.0, self.t]), 0.025, 0.5) / 0.5
        self.ct = (cs['timestamp_ns'] - segment['start_ns']) / 1000000000.0
        self.C = self.spline(self.ct)
        self.cs_temporal_response = bool(prior.get('model_CS_temporal_response', False))
        self.cs_history_metadata = None
        self.cs_past_time_moment = None
        if self.cs_temporal_response:
            if prior.get('CS_actual_history_query_root'):
                from trajectory_cs_history_response import load_history
                self.cs_history_metadata = load_history(prior, folder, cs['observation_ids'])
                query = self.cs_history_metadata['history_times_s']
                self.cs_response_weights = self.cs_history_metadata['response_weights']
                self.cs_past_time_moment = self.cs_history_metadata['past_time_moment']
            else:
                lags = np.asarray(prior['CS_response_lags_s'])
                anchor_weights = prior.get('CS_anchor_response_weights')
                self.cs_response_weights = np.asarray(anchor_weights)[cs['anchor_index']] if anchor_weights is not None else np.broadcast_to(np.asarray(prior['CS_response_weights']), (len(self.ct), len(lags)))
                query = self.ct[:, None] - lags[None]
            self.C_history = self.spline(query.clip(0.0, self.knots[-1]))
            self.cs_past_mass = ((query < 0) * self.cs_response_weights).sum(-1)
        self.anchor = f['anchors'][cs['anchor_index']].astype(float)
        self.cs_range = cs['range_m'].astype(float)
        self.bias = cs['bias_mean'].astype(float)
        self.variance = cs['bias_variance'].astype(float) + (f['cs_sigma'][:, None] * prior.get('CS_noise_sigma_multiplier', 1.0)) ** 2 + 0.15 ** 2 + prior.get('CS_additional_noise_floor_m', 0.0) ** 2
        if self.cs_temporal_response:
            self.variance += (self.cs_past_mass[:, None] * prior['CS_unknown_past_sigma_m']) ** 2
        self.pi = np.clip(f['cs_pi'].astype(float), 1e-05, 1 - 1e-05)
        self.anchor_index = cs['anchor_index'].astype(int)
        self.cs_weight = np.ones(len(self.ct))
        self.speed, self.speed_sigma, self.speed_valid = speed
        self.initial = np.asarray(segment['initial_position_xy_m'], float)
        self.n = len(self.knots)
        self.sensor_factors_only = False
        self.learned_path = learned_path
        if learned_path is not None:
            horizon = prior['learned_displacement_horizon_s']
            query = np.arange(max(self.t[0], 0.5), self.t[-1] - horizon + 1e-06, prior['learned_displacement_interval_s'])
            self.D = self.spline(query + horizon) - self.spline(query)
            self.learned_delta = np.stack([np.interp(query + horizon, self.t, learned_path[:, j]) - np.interp(query, self.t, learned_path[:, j]) for j in range(2)], -1)
            if prior.get('learned_acceleration_center', False):
                proposed_knots = np.stack([np.interp(self.knots, self.t, learned_path[:, j]) for j in range(2)], -1)
                self.learned_acceleration = self.A @ proposed_knots

    def unpack(self, vector):
        return (np.vstack((self.initial, vector[:2 * (self.n - 1)].reshape(-1, 2))), vector[-4:])

    def objective(self, vector, mode):
        factor_weights = self.prior.get('trajectory_factor_weights', {})
        q, db = self.unpack(vector)
        path = self.B @ q
        offset = self.g[None, :, :] - path[:, None, :]
        kernel = -0.5 * (offset ** 2).sum(-1) / self.bandwidth2
        numerator = kernel + self.scores[:, mode, :]
        ln, ld = (logsumexp(numerator, -1), logsumexp(kernel, -1))
        pn, pd = (np.exp(numerator - ln[:, None]), np.exp(kernel - ld[:, None]))
        eradio = -float(((ln - ld) * self.wifi_weight).sum()) * factor_weights.get('radio', 1.0)
        gp = (pd - pn) @ self.g / self.bandwidth2 * self.wifi_weight[:, None] * factor_weights.get('radio', 1.0)
        grad = self.B.T @ gp
        if self.cs_temporal_response:
            cpath = np.einsum('mln,nd->mld', self.C_history, q)
            history_d = cpath - self.anchor[:, None]
            history_norm = np.sqrt((history_d * history_d).sum(-1) + 1e-08)
            norm = (history_norm * self.cs_response_weights).sum(-1)
        else:
            cpath = self.C @ q
            d = cpath - self.anchor
            norm = np.sqrt((d * d).sum(-1) + 1e-08)
        residual = self.cs_range - (norm + self.bias[:, mode] + db[self.anchor_index])
        variance = np.maximum(self.variance[:, mode], 0.05 ** 2)
        normal = -0.5 * (residual ** 2 / variance + np.log(variance) + math.log(2 * math.pi)) + np.log1p(-self.pi)
        heavy = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi) - math.log(10.0) - 2 * np.log1p(residual ** 2 / 300) + np.log(self.pi)
        logp = np.logaddexp(normal, heavy)
        r = np.exp(normal - logp)
        dc = -(r * residual / variance + (1 - r) * 4 * residual / (300 + residual ** 2)) * self.cs_weight
        ecs = -float((logp * self.cs_weight).sum())
        if self.cs_temporal_response:
            grad += np.einsum('mln,mld,ml,m->nd', self.C_history, history_d / history_norm[..., None], np.broadcast_to(self.cs_response_weights, history_norm.shape), dc)
        else:
            grad += self.C.T @ (dc[:, None] * d / norm[:, None])
        bias_sigma = np.asarray(self.prior.get('path_CS_bias_sigma_m', 0.5))
        gb = np.bincount(self.anchor_index, weights=dc, minlength=4) + db / bias_sigma ** 2
        ebias = 0.5 * float((db ** 2 / bias_sigma ** 2).sum())
        if self.sensor_factors_only:
            return (eradio + ecs + ebias, np.r_[grad[1:].ravel(), gb])
        acc = self.A @ q
        scale2 = self.prior['acceleration_scale_m_s2'] ** 2
        if self.prior.get('learned_acceleration_center', False) and self.learned_path is not None:
            acc = acc - self.learned_acceleration
            scale2 = self.prior['learned_acceleration_sigma_m_s2'] ** 2
        a2 = (acc * acc).sum(-1)
        em = float((2.5 * np.log1p(a2 / (3 * scale2)) * self.adt).sum()) * factor_weights.get('acceleration', 1.0)
        grad += self.A.T @ (5 * acc / (3 * scale2 + a2)[:, None] * self.adt[:, None]) * factor_weights.get('acceleration', 1.0)
        vel = self.V @ q
        vnorm = np.sqrt((vel * vel).sum(-1) + 1e-06)
        excessive = np.maximum(vnorm - self.prior['source_speed_p95_m_s'], 0.0)
        if self.prior.get('learned_acceleration_center', False) and self.learned_path is not None:
            excessive = np.zeros_like(excessive)
        em += float((0.5 * (excessive / 0.5) ** 2 * self.wifi_weight).sum())
        gv = excessive / 0.5 ** 2 * self.wifi_weight
        if self.use_speed:
            residual_speed = vnorm - self.speed
            weights = self.wifi_weight * self.speed_valid
            em += float((2 * np.log1p(residual_speed ** 2 / (3 * self.speed_sigma ** 2)) * weights).sum())
            gv += 4 * residual_speed / (3 * self.speed_sigma ** 2 + residual_speed ** 2) * weights
        grad += self.V.T @ (gv[:, None] * vel / vnorm[:, None])
        if self.learned_path is not None:
            error = self.D @ q - self.learned_delta
            e2 = (error * error).sum(-1)
            sigma2 = self.prior['learned_displacement_sigma_m'] ** 2
            interval_weight = 0.5
            em += float((2.5 * np.log1p(e2 / (3 * sigma2))).sum()) * interval_weight * factor_weights.get('chord', 1.0)
            grad += self.D.T @ (5 * error / (3 * sigma2 + e2)[:, None]) * interval_weight * factor_weights.get('chord', 1.0)
        value = eradio + ecs + em + ebias
        derivative = np.r_[grad[1:].ravel(), gb]
        if not np.isfinite(value) or not np.isfinite(derivative).all():
            raise FloatingPointError('Nonfinite trajectory objective')
        return (value, derivative)

    def solve(self):
        outcomes = []
        for mode in range(8):
            xy = np.asarray([p['mode_xy_m'][mode] for p in self.predictions])
            initial = np.stack([np.interp(self.knots, self.t, xy[:, j]) for j in range(2)], -1)
            vector = np.r_[initial[1:].ravel(), np.zeros(4)]
            starts = [vector]
            if self.learned_path is not None:
                seed = np.stack([np.interp(self.knots, self.t, self.learned_path[:, j]) for j in range(2)], -1)
                seed += self.initial - seed[0]
                starts.append(np.r_[seed[1:].ravel(), np.zeros(4)])
            for vector in starts:
                result = minimize(self.objective, vector, args=(mode,), jac=True, method='L-BFGS-B', options=dict(maxiter=350, ftol=1e-09, gtol=0.0001, maxls=30))
                outcomes.append((float(result.fun), mode, result))
        _, mode, result = min(outcomes, key=lambda r: r[0])
        q, bias = self.unpack(result.x)
        xy = self.B @ q
        predictions = []
        for p, x in zip(self.predictions, xy):
            record = dict(p)
            record.update(xy_m=x.tolist(), inference_timing='OFFLINE_FULL_SEGMENT_SENSOR_FACTORS', trajectory_mode=mode, later_position_GT_used=False)
            predictions.append(record)
        return (predictions, dict(mode=mode, iterations=int(result.nit), converged=bool(result.success), termination=str(result.message), objective=float(result.fun), additional_anchor_bias_m=bias.tolist(), branch_objectives=[dict(mode=m, value=v, iterations=int(r.nit), converged=bool(r.success)) for v, m, r in outcomes]))

def job(task):
    base, capture, output, cohort, s, prior, use_speed, learned_root = task
    base, capture, output = (Path(base), Path(capture), Path(output))
    torch.set_num_threads(1)
    data = base / ('runs/ap_movement_v79_291/experiment/preprocessed' if cohort == 'AP_movement' else 'data/processed_rssi_causal_short_v1')
    source = capture / cohort / s['trajectory_id']
    target = output / cohort / s['trajectory_id']
    from support_label_barrier import deny_label_reads
    with deny_label_reads() as audit:
        with np.load(source / 'trajectory_factors.npz', allow_pickle=False) as z:
            times = z['times_ns'].copy()
        speed = speed_evidence(base, data, s, times)
        if AVAILABLE_CPUS:
            os.sched_setaffinity(0, AVAILABLE_CPUS)
        learned_path = None
        if learned_root:
            learned = json.loads((Path(learned_root) / cohort / s['trajectory_id'] / 'predictions.json').read_text())
            assert [p['timestamp_ns'] for p in learned] == times.tolist()
            learned_path = np.asarray([p['xy_m'] for p in learned], float)
        solver = SensorPath(source, s, prior, speed, use_speed, learned_path)
        predictions, diagnostic = solver.solve()
        target.mkdir(parents=True, exist_ok=True)
        write(target / 'predictions.json', predictions)
        for name in ['PRE_forecasts.npz', 'CS_diagnostics.json', 'WiFi_quality_diagnostics.json']:
            shutil.copyfile(str(source / name), str(target / name))
        diagnostic.update(trajectory_id=s['trajectory_id'], labels_opened=False, offline_future_sensors_used=True, known_start_only=True, assumed_path_shape=None, source_speed_factor=use_speed)
        diagnostic['learned_displacements_only'] = bool(learned_root)
        write(target / 'trajectory_solver.json', diagnostic)
        assert not audit['blocked_label_attempts']
    return diagnostic

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cohorts', nargs='+', default=['source_train'])
    p.add_argument('--speed', action='store_true')
    p.add_argument('--prior', type=Path)
    p.add_argument('--learned-path', type=Path)
    p.add_argument('--workers', type=int, default=24)
    a = p.parse_args()
    began = time.monotonic()
    prior = json.loads((a.prior or a.capture / 'source_prior.json').read_text())
    cohorts = json.loads((a.capture / 'segments.json').read_text())
    jobs = [(str(a.base), str(a.capture), str(a.output), c, s, prior, a.speed, str(a.learned_path) if a.learned_path else None) for c in a.cohorts for s in cohorts[c]]
    if (a.output / 'prediction_complete.json').exists():
        raise FileExistsError('Preserve the completed path experiment')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        reports = []
        for r in pool.map(job, jobs):
            reports.append(r)
            if len(reports) % 12 == 0:
                print(json.dumps(dict(completed=len(reports), total=len(jobs), seconds=time.monotonic() - began)), flush=True)
    write(a.output / 'prediction_complete.json', dict(status='COMPLETE_SENSOR_ONLY_OFFLINE_PATHS', reports=reports, source_prior=prior, labels_opened=False, seconds=time.monotonic() - began, inference_timing='OFFLINE_FULL_SEGMENT', RSSI_calibration_changed=False))
if __name__ == '__main__':
    main()
