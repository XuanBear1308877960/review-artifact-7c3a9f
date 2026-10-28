import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from engine_utils import write
from trajectory_axial_motion import feature_speeds
from trajectory_force_marginal import force_energy

def measured_intervals(evidence, minimum_span):
    token = np.asarray(evidence['motion'], float)
    t = np.asarray(evidence['times'], float)
    columns = token[:, :6].reshape(-1, 3, 2)
    R = np.concatenate((columns, np.cross(columns[..., 0], columns[..., 1])[:, :, None]), -1)
    valid = (token[:, 22] > 0.5) & (token[:, 21] * 5 <= 0.5) & (token[:, 23] > 0)
    if not valid.any():
        return (np.zeros((0, 2)), np.zeros((0, 2)))
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
    dt = token[:, 20] * 5
    left = t - dt
    if len(t) and dt[0] <= 1e-06:
        dt[0] = t[0]
        left[0] = 0.0
    intervals = []
    means = []
    group = []
    for i in range(len(t)):
        if not valid[i] or dt[i] <= 1e-05 or dt[i] > 2.0 or (left[i] < 0.5) or (t[i] > t[-1] - 0.5):
            group = []
            continue
        if group and abs(left[i] - t[group[-1]]) > 0.05:
            group = []
        group.append(i)
        span = t[i] - left[group[0]]
        if span < minimum_span:
            continue
        indices = np.asarray(group)
        coverage = float(dt[indices].sum() / span)
        if 0.9 <= coverage <= 1.1:
            intervals.append([float(left[group[0]]), float(t[i])])
            means.append((acc[indices] * dt[indices, None]).sum(0) / dt[indices].sum())
        group = []
    return (np.asarray(intervals, float).reshape(-1, 2), np.asarray(means, float).reshape(-1, 2))

def source_rows(a, span, evidence):
    source = json.loads((a.capture / 'segments.json').read_text())['source_train']
    labels = {}
    rows = []
    for ordinal, s in enumerate(source):
        name = s['device']
        if name not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target in preintegrated force training')
        if name not in labels:
            labels[name] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / name / 'evaluation_labels.json').read_text())
        pred = json.loads((a.capture / 'source_train' / s['trajectory_id'] / 'predictions.json').read_text())
        pred = [p for p in pred if p['observation_id'] in labels[name]]
        t = np.r_[0.0, [(p['timestamp_ns'] - s['start_ns']) / 1000000000.0 for p in pred]]
        xy = np.vstack((s['initial_position_xy_m'], [labels[name][p['observation_id']]['gt_xy_m'] for p in pred]))
        interval, force = measured_intervals(evidence['source_train', s['trajectory_id']], span)
        use = (interval[:, 0] >= 0.5) & (interval[:, 1] <= t[-1] - 0.5)
        interval, force = (interval[use], force[use])
        if len(force) < 5:
            continue
        grid = np.arange(0, t[-1] + 1e-05, 0.1)
        dense = np.stack([np.interp(grid, t, xy[:, j]) for j in range(2)], -1)
        smooth = gaussian_filter1d(dense, 3.5, axis=0, mode='nearest')
        spline = CubicSpline(grid, smooth, axis=0)
        width = interval[:, 1] - interval[:, 0]
        acceleration = (spline(interval[:, 1], 1) - spline(interval[:, 0], 1)) / width[:, None]
        rows.append(dict(force=force, acceleration=acceleration, weight=width, heldout=ordinal % 5 == 0))
    return rows

def fit(a):
    evidence = feature_speeds(a.features)
    fits = []
    for span in [0.5, 1.0, 2.0]:
        rows = source_rows(a, span, evidence)

        def objective(p, heldout):
            return sum((force_energy(r['force'], r['acceleration'], r['weight'], p, observation_model=True) for r in rows if r['heldout'] == heldout)) / sum((2 * len(r['force']) for r in rows if r['heldout'] == heldout))
        bounds = [(math.log(0.001), math.log(3.0)), (math.log(0.03), math.log(8.0)), (math.log(0.01), math.log(4.0))]
        result = minimize(objective, np.log([0.5, 0.5, 0.1]), args=(False,), method='L-BFGS-B', bounds=bounds)

        def null(p, heldout):
            return objective(np.r_[math.log(1e-12), p], heldout)
        reference = minimize(null, np.log([0.5, 0.1]), args=(False,), method='L-BFGS-B', bounds=bounds[1:])
        held = objective(result.x, True)
        zero = null(reference.x, True)
        samples = sum((2 * len(r['force']) for r in rows if r['heldout']))
        seconds = sum((r['weight'].sum() for r in rows if r['heldout']))
        skill = (zero - held) * samples / seconds
        row = dict(minimum_span_s=span, parameters=result.x.tolist(), heldout_NLL=held, null_heldout_NLL=zero, heldout_information_gain_nats_per_second=float(skill), source_trajectories=len(rows), heldout_force_windows=samples // 2, converged=bool(result.success))
        fits.append(row)
        print(json.dumps(row), flush=True)
    best = max(fits, key=lambda r: r['heldout_information_gain_nats_per_second'])
    prior = json.loads(a.prior.read_text())
    prior.update(marginal_force_parameters=best['parameters'], marginal_force_observation_model=True, marginal_force_selected=bool(best['heldout_information_gain_nats_per_second'] > 0), force_preintegration_minimum_span_s=best['minimum_span_s'], force_preintegration_source_fits=fits, force_preintegration_tokens_never_reused=True, force_preintegration_target_labels_used=False, force_preintegration_observation='Mean acceleration over actual nonoverlapping acquired IMU intervals; path velocity difference over same times', force_preintegration_selection='Held source likelihood-ratio information per measured second, not raw NLL across different averaging widths')
    write(a.output, prior)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'features', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    fit(p.parse_args())
if __name__ == '__main__':
    main()
