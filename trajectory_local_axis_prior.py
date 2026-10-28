import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, i0e, logsumexp
from engine_utils import write

def start_features(initial):
    x, y = np.asarray(initial, float) / np.array([10.0, 5.0])
    return np.array([1.0, x, y, x * x, x * y, y * y])

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    prior = json.loads(a.prior.read_text())
    source = json.loads((a.capture / 'segments.json').read_text())['source_train']
    labels = {}
    features = []
    angles = []
    held = []
    for i, s in enumerate(source):
        name = s['device']
        if name not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target in local source mobility fit')
        if name not in labels:
            labels[name] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / name / 'evaluation_labels.json').read_text())
        pred = json.loads((a.capture / 'source_train' / s['trajectory_id'] / 'predictions.json').read_text())
        xy = np.asarray([labels[name][r['observation_id']]['gt_xy_m'] for r in pred if r['observation_id'] in labels[name]])
        centred = xy - xy.mean(0)
        _, vectors = np.linalg.eigh(centred.T @ centred)
        axis = vectors[:, -1]
        angles.append(math.atan2(axis[1], axis[0]))
        features.append(start_features(s['initial_position_xy_m']))
        held.append(i % 5 == 0)
    X = np.asarray(features)
    theta = np.asarray(angles)
    held = np.asarray(held)
    parameters = np.asarray(prior['axial_motion_prior_parameters'])
    kappa = np.exp(parameters[2:4])
    component = kappa[None] * np.cos(2 * (theta[:, None] - parameters[None, :2])) - np.log(i0e(kappa))[None] - kappa[None] - math.log(math.pi)
    initial = np.r_[parameters[4], np.zeros(5)]

    def loss(beta, mask, regularization=0.0):
        logits = X[mask] @ beta
        w = expit(logits)
        log_weights = np.stack((-np.logaddexp(0, -logits), -np.logaddexp(0, logits)), -1)
        marginal = logsumexp(log_weights + component[mask], -1)
        responsibility = np.exp(log_weights[:, 0] + component[mask, 0] - marginal)
        value = -float(marginal.mean()) + regularization * float(np.square(beta[1:]).sum())
        gradient = X[mask].T @ (w - responsibility) / mask.sum()
        gradient[1:] += 2 * regularization * beta[1:]
        return (value, gradient)
    fits = []
    for regularization in [0.01, 0.1, 1.0]:
        fit = minimize(loss, initial, args=(~held, regularization), jac=True, method='L-BFGS-B', bounds=[(-12, 12)] * 6)
        fits.append(dict(coefficients=fit.x.tolist(), source_train_axis_NLL=loss(fit.x, ~held)[0], source_heldout_axis_NLL=loss(fit.x, held)[0], regularization=regularization))
    selected = min(fits, key=lambda r: r['source_heldout_axis_NLL'])
    baseline = loss(initial, held)[0]
    use = selected['source_heldout_axis_NLL'] < baseline
    if use:
        prior['known_start_axis_logit_coefficients'] = selected['coefficients']
    prior.update(local_axis_source_fits=fits, local_axis_selected=bool(use), local_axis_baseline_heldout_NLL=baseline, local_axis_target_GT_used=False, local_axis_known_start_only=True, local_axis_scope='Soft source-scene spatial mobility distribution, both axes retained; no phone bearing or route template')
    write(a.output, prior)
    print(json.dumps(dict(selected=use, source_heldout_axis_NLL=selected['source_heldout_axis_NLL'], baseline_heldout_NLL=baseline, coefficients=selected['coefficients'])), flush=True)
if __name__ == '__main__':
    main()
