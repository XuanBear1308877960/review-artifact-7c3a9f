import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from engine_utils import write

def density(points, centres, sigma, derivative=False):
    difference = np.asarray(points)[:, None] - np.asarray(centres)[None]
    squared = np.square(difference).sum(-1)
    denominator = 4 * sigma * sigma + squared
    lp = -math.log(2 * math.pi * sigma * sigma) - 3 * np.log1p(squared / (4 * sigma * sigma))
    total = logsumexp(lp, axis=-1)
    energy = math.log(len(centres)) - total
    if not derivative:
        return energy
    probability = np.exp(lp - total[:, None])
    gradient = (probability[..., None] * 6 * difference / denominator[..., None]).sum(1)
    return (energy, gradient)

def centres(points):
    p = np.asarray(points)
    cell = np.floor(p / 0.5).astype(int)
    unique, inv = np.unique(cell, axis=0, return_inverse=True)
    count = np.bincount(inv)
    return np.stack([np.bincount(inv, weights=p[:, j]) / count for j in range(2)], -1)

def fit(a):
    began = time.monotonic()
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    labels = {}
    train = {}
    held = []
    for ordinal, s in enumerate(segments):
        phone = s['device']
        if phone not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target route in source spatial support')
        if phone not in labels:
            labels[phone] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / phone / 'evaluation_labels.json').read_text())
        saved = json.loads((a.capture / 'source_train' / s['trajectory_id'] / 'predictions.json').read_text())
        xy = np.vstack((s['initial_position_xy_m'], [labels[phone][r['observation_id']]['gt_xy_m'] for r in saved if r['observation_id'] in labels[phone]]))
        arc = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=-1).cumsum()]
        valid = np.r_[True, np.diff(arc) > 1e-05]
        arc = arc[valid]
        xy = xy[valid]
        sample = np.stack([np.interp(np.arange(0, arc[-1], 0.5), arc, xy[:, j]) for j in range(2)], -1)
        if ordinal % 5:
            train.setdefault(phone, []).append(sample)
        else:
            held.append(sample)
    train = {p: np.concatenate(v) for p, v in train.items()}
    cross = {p: centres(np.concatenate([v for q, v in train.items() if q != p])) for p in train}

    def objective(logsigma):
        return float(np.mean([density(v, cross[p], math.exp(logsigma)).mean() for p, v in train.items()]))
    result = minimize_scalar(objective, bounds=(math.log(0.15), math.log(2.0)), method='bounded')
    sigma = math.exp(float(result.x))
    C = centres(np.concatenate(list(train.values())))
    H = np.concatenate(held)
    direction = np.asarray([[0.6, 0.8]])
    point = C[len(C) // 2:len(C) // 2 + 1] + [0.13, 0.27]
    step = 1e-05
    _, g = density(point, C, sigma, True)
    numeric = (density(point + step * direction, C, sigma) - density(point - step * direction, C, sigma)) / (2 * step)
    error = float(abs(numeric[0] - (g * direction).sum()))
    if error > 1e-06:
        raise ValueError('Spatial-support gradient mismatch')
    prior = json.loads(a.prior.read_text())
    prior.update(source_spatial_support_centres_m=C.tolist(), source_spatial_support_sigma_m=sigma, source_spatial_support_strength=0.0, source_spatial_support_train_trajectories=43, source_spatial_support_selection_trajectories=11, source_spatial_support_source_phone_crossfit=True, source_spatial_support_cross_phone_NLL=float(result.fun), source_spatial_support_heldout_NLL=float(density(H, C, sigma).mean()), source_spatial_support_gradient_difference=error, source_spatial_support_target_positions_used=False, source_spatial_support_route_time_templates_used=False, source_spatial_support_scope='Soft scene-specific source spatial-density regularizer; no walls inferred, unseen-route generalization unproven', source_spatial_support_fit_seconds=time.monotonic() - began, cache_solver_groups=True, save_solver_state=True)
    a.output.mkdir(parents=True, exist_ok=True)
    write(a.output / 'support_prior.json', prior)
    for strength in [0.1, 0.3, 1.0]:
        p = dict(prior, source_spatial_support_strength=strength)
        write(a.output / ('support_%03d_prior.json' % round(100 * strength)), p)
    print(json.dumps(dict(centres=len(C), sigma_m=sigma, cross_phone_NLL=float(result.fun), source_heldout_NLL=prior['source_spatial_support_heldout_NLL'], seconds=time.monotonic() - began)), flush=True)

def select(a):
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    held = [s for i, s in enumerate(segments) if i % 5 == 0]
    labels = {}
    scores = []
    roots = [a.baseline] + [a.output / ('held_support_%03d' % round(100 * s)) for s in a.strengths]
    for root in roots:
        if not (root / 'prediction_complete.json').exists():
            raise ValueError('Source selection path solve is incomplete: ' + str(root))
        allerrors = []
        per = {}
        for s in held:
            phone = s['device']
            if phone not in labels:
                labels[phone] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / phone / 'evaluation_labels.json').read_text())
            saved = json.loads((root / 'source_train' / s['trajectory_id'] / 'predictions.json').read_text())
            available = [r for i, r in enumerate(saved) if i >= 4 and r['observation_id'] in labels[phone]]
            error = np.asarray([np.linalg.norm(np.asarray(r['xy_m']) - labels[phone][r['observation_id']]['gt_xy_m']) for r in available])
            allerrors.extend(error.tolist())
            per[s['trajectory_id']] = float(np.median(error))
        e = np.asarray(allerrors)
        metrics = dict(n=len(e), median_m=float(np.median(e)), p90_m=float(np.percentile(e, 90)), p95_m=float(np.percentile(e, 95)), mean_m=float(np.mean(e)))
        if len(e) != 654:
            raise ValueError('Source held point count changed')
        score = metrics['median_m'] + 0.25 * metrics['mean_m'] + 0.1 * metrics['p90_m'] + 0.1 * metrics['p95_m']
        scores.append(dict(root=str(root), metrics=metrics, selection_score=score, per_trajectory_median_m=per))
    chosen = min(range(len(scores)), key=lambda i: scores[i]['selection_score'])
    strength = 0.0 if chosen == 0 else a.strengths[chosen - 1]
    prior = json.loads((a.output / 'support_prior.json').read_text())
    prior['source_spatial_support_strength'] = strength
    prior['source_spatial_support_path_selection'] = dict(scope='Actual 11 source-held trajectory paths, 654 points; no target scoring', selected_strength=strength, results=scores)
    write(a.output / 'selected_prior.json', prior)
    write(a.output / 'source_selection.json', prior['source_spatial_support_path_selection'])
    print(json.dumps(dict(selected_strength=strength, metrics=scores[chosen]['metrics'], baseline=scores[0]['metrics'])), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['fit', 'select'], required=True)
    for name in ['base', 'capture', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--strengths', type=float, nargs='+', default=[0.1, 0.3, 1.0])
    a = p.parse_args()
    fit(a) if a.stage == 'fit' else select(a)
if __name__ == '__main__':
    main()
