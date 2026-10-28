import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.special import softmax
import torch
from engine_utils import write
from trajectory_axial_motion import feature_speeds
CONST = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi)

def velocity_energy(velocity, speed, dt, parameters, temporal=True, derivative=False, speed_sigma=None):
    p = np.asarray(parameters)
    N = len(speed)
    gain = math.exp(p[3])
    means = np.zeros((N, 5, 2))
    means[:, 0, 0] = speed
    means[:, 1, 0] = -speed
    means[:, 2, 1] = gain * speed
    means[:, 3, 1] = -gain * speed
    sp, st, ss = np.exp(p[:3])
    sigma = np.asarray([[sp, st], [sp, st], [st, sp], [st, sp], [ss, ss]])
    sigma = sigma[None] / np.sqrt(np.asarray(dt))[:, None, None]
    if len(p) > 9:
        if speed_sigma is None:
            raise ValueError('Missing measured-context speed uncertainty')
        extra = np.zeros_like(sigma)
        variance = math.exp(p[9]) * np.square(speed_sigma)
        extra[:, 0:2, 0] = variance[:, None]
        extra[:, 2:4, 1] = gain * gain * variance[:, None]
        sigma = np.sqrt(sigma * sigma + extra)
    residual = np.asarray(velocity)[:, None, :] - means
    emission = (CONST - np.log(sigma) - 2 * np.log1p(residual * residual / (3 * sigma * sigma))).sum(-1)
    mass = softmax([0.0, p[4], p[5]])
    prob = np.asarray([mass[0] / 2, mass[0] / 2, mass[1] / 2, mass[1] / 2, mass[2]])
    persistence = np.exp(p[[6, 6, 7, 7, 8]])
    transition = []
    for gap in 0.5 * (dt[1:] + dt[:-1]):
        stick = np.exp(-gap / persistence) if temporal else np.zeros(5)
        transition.append((1 - stick[:, None]) * prob[None] + np.diag(stick))
    maximum = emission.max(-1)
    density = np.exp(emission - maximum[:, None])
    alpha = np.empty_like(density)
    scale = np.empty(N)
    alpha[0] = prob * density[0]
    scale[0] = alpha[0].sum()
    alpha[0] /= scale[0]
    for i in range(1, N):
        alpha[i] = alpha[i - 1] @ transition[i - 1] * density[i]
        scale[i] = alpha[i].sum()
        alpha[i] /= scale[i]
    energy = -float((np.log(scale) + maximum).sum())
    if not derivative:
        return energy
    backward = np.ones_like(alpha)
    for i in range(N - 2, -1, -1):
        backward[i] = transition[i] @ (density[i + 1] * backward[i + 1]) / scale[i + 1]
    posterior = alpha * backward
    posterior /= posterior.sum(-1, keepdims=True)
    gradient = (posterior[:, :, None] * 4 * residual / (3 * sigma * sigma + residual * residual)).sum(1)
    return (energy, gradient, posterior)

def source_rows(a):
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    features = feature_speeds(a.features)
    labels = {}
    rows = []
    for ordinal, s in enumerate(segments):
        name = s['device']
        if name not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target labels in motion-state fitting')
        if name not in labels:
            labels[name] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / name / 'evaluation_labels.json').read_text())
        folder = a.capture / 'source_train' / s['trajectory_id']
        saved = json.loads((folder / 'predictions.json').read_text())
        selected = [p for p in saved if p['observation_id'] in labels[name]]
        t = np.r_[0.0, [(p['timestamp_ns'] - s['start_ns']) / 1000000000.0 for p in selected]]
        xy = np.vstack((s['initial_position_xy_m'], [labels[name][p['observation_id']]['gt_xy_m'] for p in selected]))
        with np.load(folder / 'CS_training_contexts.npz', allow_pickle=False) as z:
            cs_end = (z['timestamp_ns'][-1] - s['start_ns']) / 1000000000.0
        end = max(t[-1], cs_end)
        knots = np.linspace(0.0, end, int(np.ceil(end / 0.5)) + 1)
        mid = 0.5 * (knots[1:] + knots[:-1])
        dt = np.diff(knots)
        grid = np.arange(0, t[-1] + 1e-05, 0.1)
        dense = np.stack([np.interp(grid, t, xy[:, j]) for j in range(2)], -1)
        curve = CubicSpline(grid, gaussian_filter1d(dense, 2.5, axis=0, mode='nearest'), axis=0)
        _, vec = np.linalg.eigh(np.cov(xy.T))
        axis = vec[:, -1]
        normal = np.array([-axis[1], axis[0]])
        velocity = curve(mid, 1) @ np.stack((axis, normal), -1)
        valid = (mid >= 0.5) & (mid <= t[-1] - 0.5) & (np.linalg.norm(velocity, axis=-1) < 4.0)
        f = features['source_train', s['trajectory_id']]
        speed = np.interp(mid, f['times'], f['speed']).clip(0.1, 3.0)
        if valid.sum() < 5:
            continue
        uncertainty = np.interp(mid, f['times'], f['sigma'])
        rows.append(dict(velocity=velocity[valid], speed=speed[valid], speed_sigma=uncertainty[valid], dt=dt[valid], time_s=mid[valid], heldout=ordinal % 5 == 0, trajectory_id=s['trajectory_id'], full_time_s=mid, full_dt=dt, valid_indices=np.flatnonzero(valid), full_speed=speed, full_speed_sigma=uncertainty))
    return rows

def fit(a):
    torch.set_num_threads(1)
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    rows = source_rows(a)
    N = len(rows)
    M = max((len(r['speed']) for r in rows))
    velocity = np.zeros((N, M, 2))
    speed = np.zeros((N, M))
    dt = np.ones((N, M)) * 0.5
    valid = np.zeros((N, M), bool)
    for i, r in enumerate(rows):
        n = len(r['speed'])
        velocity[i, :n] = r['velocity']
        speed[i, :n] = r['speed']
        dt[i, :n] = r['dt']
        valid[i, :n] = True
    vel, sp, step, mask = [torch.tensor(x, dtype=torch.float64) for x in [velocity, speed, dt, valid]]
    uncertainty = np.ones((N, M)) * 0.5
    for i, r in enumerate(rows):
        uncertainty[i, :len(r['speed'])] = r['speed_sigma']
    speed_sigma = torch.tensor(uncertainty, dtype=torch.float64)
    held = torch.tensor([r['heldout'] for r in rows])
    train = ~held
    initial = np.r_[np.log([0.35, 0.18, 0.22, 0.8]), -2.0, -3.0, np.log([2.0, 0.4, 0.5])]
    lower = np.r_[np.log([0.08, 0.06, 0.08, 0.3]), -6.0, -6.0, np.log([0.05, 0.05, 0.05])]
    upper = np.r_[np.log([2.0, 2.0, 2.0, 1.5]), 2.0, 2.0, np.log([20.0, 10.0, 10.0])]
    if a.use_speed_uncertainty:
        initial = np.r_[prior['velocity_state_parameters'][:9], -3.0]
        lower = np.r_[lower, math.log(0.0001)]
        upper = np.r_[upper, math.log(3.0)]

    def likelihood(p, temporal):
        zero = torch.zeros_like(sp)
        gain = p[3].exp()
        means = torch.stack((torch.stack((sp, zero), -1), torch.stack((-sp, zero), -1), torch.stack((zero, gain * sp), -1), torch.stack((zero, -gain * sp), -1), torch.stack((zero, zero), -1)), 2)
        e = vel[:, :, None, :] - means
        s1, s2, ss = p[:3].exp()
        sig = torch.stack((torch.stack((s1, s2)), torch.stack((s1, s2)), torch.stack((s2, s1)), torch.stack((s2, s1)), torch.stack((ss, ss))))
        sig = sig[None, None] / step[:, :, None, None].sqrt()
        if len(p) > 9:
            variance = p[9].exp() * speed_sigma.square()
            extra = torch.stack((torch.stack((variance, zero), -1), torch.stack((variance, zero), -1), torch.stack((zero, gain * gain * variance), -1), torch.stack((zero, gain * gain * variance), -1), torch.stack((zero, zero), -1)), 2)
            sig = (sig.square() + extra).sqrt()
        emission = (CONST - sig.log() - 2 * torch.log1p(e.square() / (3 * sig.square()))).sum(-1)
        mass = torch.softmax(torch.stack((p[4] * 0, p[4], p[5])), 0)
        prob = torch.stack((mass[0] / 2, mass[0] / 2, mass[1] / 2, mass[1] / 2, mass[2]))
        tau = p[torch.tensor([6, 6, 7, 7, 8])].exp()
        alpha = prob.log()[None] + emission[:, 0]
        for j in range(1, M):
            if temporal:
                gap = 0.5 * (step[:, j] + step[:, j - 1])
                stick = torch.exp(-gap[:, None] / tau[None])
                T = (1 - stick[:, :, None]) * prob[None, None] + torch.diag_embed(stick)
                proposed = torch.logsumexp(alpha[:, :, None] + T.clamp_min(1e-15).log(), 1) + emission[:, j]
            else:
                proposed = torch.logsumexp(alpha, -1)[:, None] + prob.log()[None] + emission[:, j]
            alpha = torch.where(mask[:, j, None].bool(), proposed, alpha)
        return -torch.logsumexp(alpha, -1)
    results = {}
    for temporal in [False, True]:
        parameter = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float64))
        optimizer = torch.optim.Adam([parameter], lr=0.025)
        best = None
        trace = []
        for update in range(401):
            loss = likelihood(parameter, temporal)
            if update % 25 == 0:
                val = float(loss[held].sum() / mask[held].sum())
                row = dict(update=update, source_heldout_NLL=val, source_train_NLL=float(loss[train].sum() / mask[train].sum()))
                trace.append(row)
                if best is None or val < best[0]:
                    best = (val, parameter.detach().numpy().copy(), update)
            if update == 400:
                break
            objective = loss[train].sum() / mask[train].sum()
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            with torch.no_grad():
                parameter.clamp_(torch.tensor(lower), torch.tensor(upper))
        results['temporal' if temporal else 'independent'] = dict(heldout_NLL=best[0], parameters=best[1].tolist(), selected_update=best[2], trace=trace)
    name = min(results, key=lambda k: results[k]['heldout_NLL'])
    selected = results[name]
    p = np.asarray(selected['parameters'])
    baseline = None
    uncertainty_selected = False
    if a.use_speed_uncertainty:
        old = prior['velocity_state_parameters']
        old_temporal = prior['velocity_state_temporal_selected']
        baseline = sum((velocity_energy(r['velocity'], r['speed'], r['dt'], old, old_temporal, speed_sigma=r['speed_sigma']) for r in rows if r['heldout'])) / sum((len(r['speed']) for r in rows if r['heldout']))
        uncertainty_selected = selected['heldout_NLL'] < baseline
        if not uncertainty_selected:
            p = np.asarray(old)
            name = 'temporal' if old_temporal else 'independent'
    r = rows[0]
    direction = np.random.RandomState(82361).normal(size=r['velocity'].shape)
    direction /= np.linalg.norm(direction)
    h = 1e-05
    _, gradient, _ = velocity_energy(r['velocity'], r['speed'], r['dt'], p, name == 'temporal', True, r['speed_sigma'])
    numeric = (velocity_energy(r['velocity'] + h * direction, r['speed'], r['dt'], p, name == 'temporal', speed_sigma=r['speed_sigma']) - velocity_energy(r['velocity'] - h * direction, r['speed'], r['dt'], p, name == 'temporal', speed_sigma=r['speed_sigma'])) / (2 * h)
    if abs(numeric - (gradient * direction).sum()) > 1e-05:
        raise ValueError('Source motion-state derivative mismatch')
    actual = sum((velocity_energy(r['velocity'], r['speed'], r['dt'], p, name == 'temporal', speed_sigma=r['speed_sigma']) for r in rows if r['heldout'])) / sum((len(r['speed']) for r in rows if r['heldout']))
    if (not a.use_speed_uncertainty or uncertainty_selected) and abs(actual - selected['heldout_NLL']) > 1e-06:
        raise ValueError('Fitted and runtime motion-state likelihood differ')
    prior.update(velocity_state_parameters=p.tolist(), velocity_state_temporal_selected=name == 'temporal', velocity_state_source_fits=results, velocity_state_target_labels_used=False, velocity_state_source_trajectories=len(rows), velocity_state_names=['along_positive', 'along_negative', 'transverse_positive', 'transverse_negative', 'stop'], velocity_state_phone_map_bearing_used=False, velocity_state_route_templates_used=False, velocity_state_fit_seconds=time.monotonic() - began)
    if a.use_speed_uncertainty:
        prior.update(velocity_state_speed_uncertainty_selected=bool(uncertainty_selected), velocity_state_constant_uncertainty_source_heldout_NLL=baseline)
    write(a.output, prior)
    print(json.dumps(dict(selected=name, heldout_NLL=actual, other_heldout_NLL={k: v['heldout_NLL'] for k, v in results.items()}, sigma=np.exp(p[:3]).tolist(), transverse_speed_gain=float(np.exp(p[3])), state_group_probability=softmax([0.0, p[4], p[5]]).tolist(), persistence_s=np.exp(p[6:9]).tolist(), uncertainty_coefficient=float(np.exp(p[9])) if len(p) > 9 else None, constant_uncertainty_heldout_NLL=baseline, seconds=time.monotonic() - began)), flush=True)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'features', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--use-speed-uncertainty', action='store_true')
    fit(p.parse_args())
if __name__ == '__main__':
    main()
