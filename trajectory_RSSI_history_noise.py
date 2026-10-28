import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
import torch
from base_evaluate import AVAILABLE_CPUS
from engine_utils import write
from trajectory_joint_calibration import field
from trajectory_joint_marginal import initial_prior, mixture_posterior
from trajectory_temporal_radio import ObservationModel, TemporalJointPath

def features(model):
    t = np.asarray(model.query_metadata['times_ns'], np.int64)
    t = (t - t[0]) / 1000000000.0
    if np.any(np.diff(t) <= 0):
        raise ValueError('RSSI quality requires increasing real timestamps')
    y = model.pre['rssi']
    mask = model.pre['mask']
    jitter = np.zeros_like(y, float)
    available = np.zeros_like(mask, bool)
    history = [[] for _ in range(4)]
    for i, now in enumerate(t):
        for ap in range(4):
            old = history[ap]
            if len(old) >= 3:
                a, b, c = old[-3:]
                dt1 = t[c] - t[b]
                dt2 = t[b] - t[a]
                if now - t[c] <= 1.0 and 0.001 < dt1 <= 1.0 and (0.001 < dt2 <= 1.0) and (0.25 <= dt1 / dt2 <= 4.0):
                    ratio = dt1 / dt2
                    innovation = y[c, ap] - (1 + ratio) * y[b, ap] + ratio * y[a, ap]
                    variance = innovation ** 2 / (1 + (1 + ratio) ** 2 + ratio ** 2)
                    jitter[i, ap] = np.log1p(min(variance / 4.0, 25.0))
                    available[i, ap] = True
            if mask[i, ap] and np.isfinite(y[i, ap]):
                old.append(i)
    return (jitter, ~available)

def covariance_with_history(model, base, parameters, noise_parameters, feature=None):
    jitter, missing = features(model) if feature is None else feature
    p = np.asarray(parameters, float)
    factor = np.exp(p[0] + p[1] * jitter + p[2] * missing)
    out = []
    for ap, C in enumerate(base):
        out.append(C + np.diag(math.exp(2 * noise_parameters[ap]) * (factor[:, ap] - 1)))
    return (out, factor)

def raw_features(model):
    q = model.query_metadata
    out = []
    for ap in range(4):
        jitter = q['raw_jitter_log_' + str(ap)]
        valid = q['raw_quality_valid_' + str(ap)]
        if jitter.shape != (model.S[ap].shape[1],) or valid.shape != jitter.shape:
            raise ValueError('Raw RSSI quality and contributor columns differ')
        out.append((jitter, ~valid))
    return out

def covariance_with_raw_history(model, base, parameters, noise_parameters, feature=None):
    quality = raw_features(model) if feature is None else feature
    p = np.asarray(parameters, float)
    out = []
    factors = []
    delta = []
    for ap, (jitter, missing) in enumerate(quality):
        factor = np.exp(p[0] + p[1] * jitter + p[2] * missing)
        S = model.S[ap]
        change = math.exp(2 * noise_parameters[9 + ap]) * (S * (factor - 1)[None]) @ S.T
        out.append(base[ap] + change)
        factors.append(factor)
        delta.append(change)
    return (out, factors, delta)

class HistoryRSSINoisePath(TemporalJointPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any((self.prior.get(k) for k in ['known_start_radio_conditioning', 'RSSI_segment_scale_marginalization', 'RSSI_AP_scale_marginalization', 'receiver_curvature_sigma_db', 'RSSI_map_variance_covariance', 'RSSI_spatial_residual_covariance'])):
            raise ValueError('Evaluate predictable RSSI variance separately from alternative covariance models')
        if not np.array_equal(self.observations.query_metadata['times_ns'], self.times_ns):
            raise ValueError('RSSI quality clock differs')
        self.radio_covariances, self.history_noise_factor = covariance_with_history(self.observations, self.radio_covariances, self.prior['RSSI_history_noise_parameters'], np.asarray(self.prior['temporal_RSSI_parameters']))
        self.radio_blocks = self.observations.blocks(self.radio_covariances)

    def calibrate_solution(self, predictions, diagnostic):
        predictions, diagnostic = super().calibrate_solution(predictions, diagnostic)
        jitter, missing = features(self.observations)
        selected = self.observations.mask
        scale = np.sqrt(self.history_noise_factor[selected])
        diagnostic.update(RSSI_history_noise_current_value_used=False, RSSI_history_noise_later_positions_used=False, RSSI_history_noise_white_sigma_multiplier_quantiles=np.percentile(scale, [0, 5, 50, 95, 100]).tolist(), RSSI_history_noise_multiplier_over4_fraction=float((scale > 4).mean()), RSSI_history_noise_missing_fraction=float(missing[selected].mean()), RSSI_history_noise_scope='Conditional on strictly past observed RSSI and existing observed median selection; colored/shared covariance unchanged')
        return (predictions, diagnostic)

class RawHistoryRSSINoisePath(TemporalJointPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any((self.prior.get(k) for k in ['known_start_radio_conditioning', 'RSSI_segment_scale_marginalization', 'RSSI_AP_scale_marginalization', 'receiver_curvature_sigma_db', 'RSSI_history_noise_source_selected', 'RSSI_map_variance_covariance', 'RSSI_spatial_residual_covariance'])):
            raise ValueError('Evaluate raw-sample RSSI quality separately from alternative covariance models')
        if not np.array_equal(self.observations.query_metadata['times_ns'], self.times_ns):
            raise ValueError('Raw quality clock differs')
        self.radio_covariances, self.raw_history_factor, delta = covariance_with_raw_history(self.observations, self.radio_covariances, self.prior['RSSI_raw_history_noise_parameters'], np.asarray(self.prior['temporal_RSSI_parameters']))
        self.radio_shared_covariances = [C + D for C, D in zip(self.radio_shared_covariances, delta)]
        self.radio_blocks = self.observations.blocks(self.radio_covariances)

    def calibrate_solution(self, predictions, diagnostic):
        predictions, diagnostic = super().calibrate_solution(predictions, diagnostic)
        selected = []
        missing = []
        quality = raw_features(self.observations)
        for ap, factor in enumerate(self.raw_history_factor):
            use = self.observations.S[ap][self.observations.mask[:, ap]].sum(0) > 0
            selected.extend(np.sqrt(factor[use]))
            missing.extend(quality[ap][1][use])
        scale = np.asarray(selected)
        diagnostic.update(RSSI_raw_history_noise_current_value_used=False, RSSI_raw_history_noise_later_positions_used=False, RSSI_raw_history_sigma_multiplier_quantiles=np.percentile(scale, [0, 5, 50, 95, 100]).tolist(), RSSI_raw_history_sigma_over4_fraction=float((scale > 4).mean()), RSSI_raw_history_quality_missing_fraction=float(np.mean(missing)), RSSI_raw_history_noise_white_floor_changed=False, RSSI_raw_history_noise_scope='Predictable raw-sample variance before shared median operator; original measurements and query geometry retained')
        return (predictions, diagnostic)

def prepare(task):
    base, capture, queries, prior, s, ordinal = task[:6]
    raw_mode = bool(task[6]) if len(task) > 6 else False
    base, capture, queries = map(Path, [base, capture, queries])
    torch.set_num_threads(1)
    if AVAILABLE_CPUS:
        os.sched_setaffinity(0, AVAILABLE_CPUS)
    phone = s['device']
    if phone not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
        raise ValueError('Target in RSSI history-noise fit')
    labels = json.loads((base / 'data/processed_rssi_causal_short_v1' / phone / 'evaluation_labels.json').read_text())
    folder = capture / 'source_train' / s['trajectory_id']
    with np.load(folder / 'PRE_forecasts.npz', allow_pickle=False) as z:
        pre = {k: z[k].copy() for k in z.files}
    pred = json.loads((folder / 'predictions.json').read_text())
    valid = [p for p in pred if p['observation_id'] in labels]
    t = np.r_[0.0, [(p['timestamp_ns'] - s['start_ns']) / 1000000000.0 for p in valid]]
    xy = np.vstack((s['initial_position_xy_m'], [labels[p['observation_id']]['gt_xy_m'] for p in valid]))
    model = ObservationModel(queries / 'source_train' / s['trajectory_id'] / 'radio_queries.npz', pre)
    quality = raw_features(model) if raw_mode else features(model)
    model.mask &= np.asarray([str(oid) in labels for oid in pre['observation_ids']])[:, None]
    raw_xy = np.stack([np.interp(model.raw_times.clip(0, t[-1]), t, xy[:, j]) for j in range(2)], -1)
    mean, B = model.geometry(raw_xy, field(base, prior))
    noise = np.asarray(prior['temporal_RSSI_parameters'])
    C, _, _ = model.noise(noise)
    return dict(model=model, quality=quality, mean=mean, basis=B, initial=initial_prior(pre), covariance=C, noise_parameters=noise, heldout=ordinal % 5 == 0, raw_mode=raw_mode)

def fit(a):
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    tasks = [(str(a.base), str(a.capture), str(a.queries), prior, s, i, a.raw_contributors) for i, s in enumerate(segments)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(prepare, tasks))

    def objective(p, heldout):
        total = 0.0
        count = 0
        for r in rows:
            if r['heldout'] != heldout:
                continue
            if r['raw_mode']:
                C, _, _ = covariance_with_raw_history(r['model'], r['covariance'], p, r['noise_parameters'], r['quality'])
            else:
                C, _ = covariance_with_history(r['model'], r['covariance'], p, r['noise_parameters'], r['quality'])
            blocks = r['model'].blocks(C)
            z, A, ld = r['model'].whiten(r['mean'], r['basis'], blocks)
            total += mixture_posterior(z, A, r['initial'], ld)[0]
            count += len(z)
        return float(total / count)
    initial = np.zeros(3)
    baseline = objective(initial, True)
    result = minimize(objective, initial, args=(False,), method='L-BFGS-B', bounds=[(-1.4, 1.4), (0.0, 2.0), (-2.0, 2.0)], options=dict(maxiter=60, ftol=1e-08))
    val = objective(result.x, True)
    selected = bool(val < baseline)
    prefix = 'RSSI_raw_history_noise' if a.raw_contributors else 'RSSI_history_noise'
    prior.update({prefix + '_parameters': (result.x if selected else initial).tolist(), prefix + '_source_selected': selected, prefix + '_source_train_NLL': float(result.fun), prefix + '_source_heldout_NLL': val, prefix + '_baseline_source_heldout_NLL': baseline, prefix + '_source_only': True, prefix + '_target_data_used': False, prefix + '_current_RSSI_used': False, prefix + '_source_fit_seconds': time.monotonic() - began, prefix + '_mean_map_changed': False, prefix + '_original_initial_eight_unchanged': True})
    prior.update(cache_solver_groups=True, save_solver_state=True)
    if a.raw_contributors and selected:
        prior['temporal_RSSI_query_root'] = str(a.queries.resolve())
    write(a.output / 'source_prior.json', prior)
    print(json.dumps(dict(selected=selected, parameters=result.x.tolist(), baseline=baseline, source_heldout_NLL=val, seconds=time.monotonic() - began)), flush=True)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'queries', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=54)
    p.add_argument('--raw-contributors', action='store_true')
    fit(p.parse_args())
if __name__ == '__main__':
    main()
