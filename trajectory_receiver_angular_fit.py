import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from scipy.linalg import solve_triangular, cholesky
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import torch
from base_evaluate import AVAILABLE_CPUS
from engine_utils import write
from trajectory_joint_calibration import field
from trajectory_joint_marginal import initial_prior, posterior
from trajectory_temporal_radio import ObservationModel

def raw_feature(model, xy, centres, initial, derivative=False):
    q = model.query_metadata
    u = np.concatenate([q['raw_attitude_horizontal_' + str(ap)] for ap in range(4)])
    valid = np.concatenate([q['raw_attitude_valid_' + str(ap)] for ap in range(4)])
    u0 = q['attitude_start_horizontal']
    d = np.asarray(centres)[model.ap] - xy
    r = np.sqrt(np.square(d).sum(-1) + 1.0)
    d0 = np.asarray(centres)[model.ap] - np.asarray(initial)
    r0 = np.sqrt(np.square(d0).sum(-1) + 1.0)
    rotate = np.stack((u, np.stack((u[:, 1], -u[:, 0]), -1)), 1)
    rotate0 = np.stack((u0, np.asarray([u0[1], -u0[0]])))
    phi = np.einsum('mcz,mz->mc', rotate, d / r[:, None]) - np.einsum('cz,mz->mc', rotate0, d0 / r0[:, None])
    phi[~valid] = 0.0
    if not derivative:
        return phi
    db = -np.eye(2)[None] / r[:, None, None] + d[:, :, None] * d[:, None, :] / r[:, None, None] ** 3
    grad = np.einsum('mcz,mzv->mcv', rotate, db)
    grad[~valid] = 0.0
    return (phi, grad)

def feature(model, xy, radio, initial, derivative=False):
    out = raw_feature(model, xy, radio.centres, initial, derivative)
    raw = out[0] if derivative else out
    value = np.zeros((model.N, 4, 2))
    for ap in range(4):
        value[:, ap] = model.S[ap] @ raw[model.bounds[ap]:model.bounds[ap + 1]]
    return (value, out[1]) if derivative else value

def prepare_one(task):
    base, capture, queries, prior, s, ordinal = task
    base, capture, queries = map(Path, [base, capture, queries])
    torch.set_num_threads(1)
    if AVAILABLE_CPUS:
        os.sched_setaffinity(0, AVAILABLE_CPUS)
    phone = s['device']
    if phone not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
        raise ValueError('Target in angular-response fit')
    labels = json.loads((base / 'data/processed_rssi_causal_short_v1' / phone / 'evaluation_labels.json').read_text())
    folder = capture / 'source_train' / s['trajectory_id']
    with np.load(folder / 'PRE_forecasts.npz', allow_pickle=False) as z:
        pre = {k: z[k].copy() for k in z.files}
    pred = json.loads((folder / 'predictions.json').read_text())
    selected = [p for p in pred if p['observation_id'] in labels]
    t = np.r_[0.0, [(p['timestamp_ns'] - s['start_ns']) / 1000000000.0 for p in selected]]
    xy = np.vstack((s['initial_position_xy_m'], [labels[p['observation_id']]['gt_xy_m'] for p in selected]))
    model = ObservationModel(queries / 'source_train' / s['trajectory_id'] / 'radio_queries.npz', pre)
    model.mask &= np.asarray([str(oid) in labels for oid in pre['observation_ids']])[:, None]
    raw_xy = np.stack([np.interp(model.raw_times.clip(0, t[-1]), t, xy[:, j]) for j in range(2)], -1)
    radio = field(base, prior)
    mean, basis = model.geometry(raw_xy, radio)
    phi = feature(model, raw_xy, radio, s['initial_position_xy_m'])
    cov, _, _ = model.noise(np.asarray(prior['temporal_RSSI_parameters']))
    blocks = model.blocks(cov)
    z, A, ld = model.whiten(mean, basis, blocks)
    V = np.concatenate([solve_triangular(L, phi[ix, ap], lower=True, check_finite=False) for ap, ix, L in blocks])
    device = initial_prior(pre)
    values = []
    for k in range(8):
        nll, eta, P, residual = posterior(z, A, device, k, ld)
        AV = A.T @ V
        B = V.T @ V - AV.T @ P @ AV
        B = (B + B.T) / 2
        h = V.T @ residual
        if np.linalg.eigvalsh(B).min() < -1e-07:
            raise ValueError('Negative angular-response precision')
        values.append(dict(nll=nll, B=B.tolist(), h=h.tolist()))
    dense_difference = None
    gradient_difference = None
    phase_difference = None
    if ordinal == 0:
        sigma = 3.0
        K = np.eye(len(z)) + A @ device['covariance'][0] @ A.T + sigma * sigma * V @ V.T
        L = cholesky(K, lower=True, check_finite=False)
        white = solve_triangular(L, z - A @ device['mean'][0], lower=True, check_finite=False)
        dense = 0.5 * float(white @ white + 2 * np.log(np.diag(L)).sum() + ld + len(z) * math.log(2 * math.pi))
        v = values[0]
        D = np.eye(2) + sigma * sigma * np.asarray(v['B'])
        h = np.asarray(v['h'])
        rank = v['nll'] + 0.5 * np.linalg.slogdet(D)[1] - 0.5 * sigma * sigma * h @ np.linalg.solve(D, h)
        dense_difference = float(abs(dense - rank))
        _, g = raw_feature(model, raw_xy, radio.centres, s['initial_position_xy_m'], True)
        direction = np.random.RandomState(1948).normal(size=raw_xy.shape)
        eps = 1e-05
        numeric = (raw_feature(model, raw_xy + eps * direction, radio.centres, s['initial_position_xy_m']) - raw_feature(model, raw_xy - eps * direction, radio.centres, s['initial_position_xy_m'])) / (2 * eps)
        gradient_difference = float(abs(numeric - np.einsum('mcz,mz->mc', g, direction)).max())
        original = model.query_metadata
        model.query_metadata = {k: x.copy() for k, x in original.items()}
        theta = 0.873
        R = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        for ap in range(4):
            model.query_metadata['raw_attitude_horizontal_' + str(ap)] = original['raw_attitude_horizontal_' + str(ap)] @ R.T
        model.query_metadata['attitude_start_horizontal'] = R @ original['attitude_start_horizontal']
        rotated = feature(model, raw_xy, radio, s['initial_position_xy_m'])
        model.query_metadata = original
        phase_difference = float(abs(np.einsum('mac,nbc->manb', phi, phi) - np.einsum('mac,nbc->manb', rotated, rotated)).max())
        if dense_difference > 1e-07 or gradient_difference > 1e-06 or phase_difference > 1e-10:
            raise ValueError('Angular-response integral/gradient/phase proof failed')
    return dict(trajectory_id=s['trajectory_id'], device=phone, heldout=ordinal % 5 == 0, n=len(z), values=values, log_weights=device['log_weight'].tolist(), feature_rms=float(np.sqrt(np.mean(phi ** 2))), dense_Gaussian_difference=dense_difference, feature_gradient_difference=gradient_difference, arbitrary_yaw_covariance_difference=phase_difference)

def nll(row, sigma):
    s2 = float(sigma) ** 2
    v = row['values']
    B = np.asarray([x['B'] for x in v])
    h = np.asarray([x['h'] for x in v])
    D = np.eye(2)[None] + s2 * B
    value = np.asarray([x['nll'] for x in v]) + 0.5 * np.linalg.slogdet(D)[1] - 0.5 * s2 * np.einsum('kc,kc->k', h, np.linalg.solve(D, h))
    return -float(logsumexp(np.asarray(row['log_weights']) - value))

def main():
    p = argparse.ArgumentParser()
    for k in ['base', 'capture', 'queries', 'prior', 'output']:
        p.add_argument('--' + k, type=Path, required=True)
    p.add_argument('--workers', type=int, default=48)
    a = p.parse_args()
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    tasks = [(str(a.base), str(a.capture), str(a.queries), prior, s, i) for i, s in enumerate(segments)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(prepare_one, tasks))

    def evaluate(sigma, heldout, balanced=False):
        selected = [r for r in rows if r['heldout'] == heldout]
        if balanced:
            return float(np.mean([sum((nll(r, sigma) for r in selected if r['device'] == phone)) / sum((r['n'] for r in selected if r['device'] == phone)) for phone in sorted({r['device'] for r in selected})]))
        return float(sum((nll(r, sigma) for r in selected)) / sum((r['n'] for r in selected)))
    fit = minimize_scalar(lambda log: evaluate(math.exp(log), False, True), bounds=(math.log(0.01), math.log(20.0)), method='bounded', options=dict(xatol=1e-05))
    sigma = math.exp(fit.x)
    baseline = evaluate(0.0, True)
    candidate = evaluate(sigma, True)
    selected = sigma if candidate < baseline else 0.0
    result = dict(status='COMPLETE_SOURCE_ANGULAR_RESPONSE_PROPOSAL_NOT_NATIVE_LOCALIZATION', source_prior=prior, source_trajectories=54, source_train_trajectories=43, source_heldout_trajectories=11, source_heldout_baseline_NLL=baseline, source_heldout_angular_NLL=candidate, source_train_baseline_NLL=evaluate(0.0, False), source_train_angular_NLL=evaluate(sigma, False), fitted_sigma_db=sigma, source_selected_sigma_db=selected, target_data_used=False, original_eight_six_dimensional_initial_components_unchanged=True, mean_map_changed=False, unknown_game_to_map_yaw_integrated_by_isotropic_pair=True, phone_attitude_is_walking_direction=False, quaternion_sensor_timestamp_available=False, rows=rows, seconds=time.monotonic() - began)
    write(a.output, result)
    print(json.dumps({k: result[k] for k in ['status', 'source_heldout_baseline_NLL', 'source_heldout_angular_NLL', 'fitted_sigma_db', 'source_selected_sigma_db', 'seconds']}), flush=True)
if __name__ == '__main__':
    main()
