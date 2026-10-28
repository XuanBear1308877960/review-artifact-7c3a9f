import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from engine_utils import write
from trajectory_velocity_states import CONST, source_rows, velocity_energy
NODES, WEIGHTS = np.polynomial.hermite.hermgauss(9)
LOG_WEIGHTS = np.log(WEIGHTS) - 0.5 * math.log(math.pi)

def gain_components(mean, sigma):
    return np.exp(mean + math.sqrt(2) * sigma * NODES)

def gain_energy(velocity, speed, dt, parameters, temporal, mean, sigma, derivative=False):
    p = np.asarray(parameters, float)
    if len(p) != 9:
        raise ValueError('Motion gain experiment uses the frozen nine-parameter velocity-state model')
    gain = gain_components(mean, sigma)
    Q = len(gain)
    N = len(speed)
    transverse = math.exp(p[3])
    scaled = gain[:, None] * np.asarray(speed)[None]
    means = np.zeros((Q, N, 5, 2))
    means[:, :, 0, 0] = scaled
    means[:, :, 1, 0] = -scaled
    means[:, :, 2, 1] = transverse * scaled
    means[:, :, 3, 1] = -transverse * scaled
    sp, st, ss = np.exp(p[:3])
    scales = np.asarray([[sp, st], [sp, st], [st, sp], [st, sp], [ss, ss]])
    scales = scales[None, None] / np.sqrt(np.asarray(dt))[None, :, None, None]
    residual = np.asarray(velocity)[None, :, None] - means
    emission = (CONST - np.log(scales) - 2 * np.log1p(residual ** 2 / (3 * scales ** 2))).sum(-1)
    mass = softmax([0.0, p[4], p[5]])
    prob = np.asarray([mass[0] / 2, mass[0] / 2, mass[1] / 2, mass[1] / 2, mass[2]])
    persistence = np.exp(p[[6, 6, 7, 7, 8]])
    transition = []
    for gap in 0.5 * (dt[1:] + dt[:-1]):
        stick = np.exp(-gap / persistence) if temporal else np.zeros(5)
        transition.append((1 - stick[:, None]) * prob[None] + np.diag(stick))
    maximum = emission.max(-1)
    density = np.exp(emission - maximum[:, :, None])
    alpha = np.empty_like(density)
    scale = np.empty((Q, N))
    alpha[:, 0] = prob[None] * density[:, 0]
    scale[:, 0] = alpha[:, 0].sum(-1)
    alpha[:, 0] /= scale[:, 0, None]
    for i in range(1, N):
        alpha[:, i] = alpha[:, i - 1] @ transition[i - 1] * density[:, i]
        scale[:, i] = alpha[:, i].sum(-1)
        alpha[:, i] /= scale[:, i, None]
    component_energy = -(np.log(scale) + maximum).sum(-1)
    logmass = LOG_WEIGHTS - component_energy
    normalizer = logsumexp(logmass)
    if not derivative:
        return -float(normalizer)
    backward = np.ones_like(alpha)
    for i in range(N - 2, -1, -1):
        backward[:, i] = density[:, i + 1] * backward[:, i + 1] @ transition[i].T / scale[:, i + 1, None]
    posterior = alpha * backward
    posterior /= posterior.sum(-1, keepdims=True)
    component_gradient = (posterior[:, :, :, None] * 4 * residual / (3 * scales ** 2 + residual ** 2)).sum(2)
    weight = np.exp(logmass - normalizer)
    gradient = (weight[:, None, None] * component_gradient).sum(0)
    return (-float(normalizer), gradient, weight)

def fit(a):
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    rows = source_rows(a)
    p = prior['velocity_state_parameters']
    temporal = prior['velocity_state_temporal_selected']

    def objective(value, heldout):
        selected = [r for r in rows if r['heldout'] == heldout]
        return sum((gain_energy(r['velocity'], r['speed'], r['dt'], p, temporal, float(value[0]), float(value[1])) for r in selected)) / sum((len(r['speed']) for r in selected))
    old = objective([0.0, 0.0], True)
    result = minimize(objective, [0.0, 0.12], args=(False,), bounds=[(-0.45, 0.4), (0.0001, 0.5)], method='L-BFGS-B', options=dict(maxiter=60, ftol=1e-08))
    if not np.isfinite(result.fun):
        raise FloatingPointError('Nonfinite source motion-scale prior')
    validation = objective(result.x, True)
    selected = validation < old
    r = rows[0]
    h = 1e-05
    direction = np.random.RandomState(82581).normal(size=r['velocity'].shape)
    direction /= np.linalg.norm(direction)
    _, g, _ = gain_energy(r['velocity'], r['speed'], r['dt'], p, temporal, *result.x, derivative=True)
    numeric = (gain_energy(r['velocity'] + h * direction, r['speed'], r['dt'], p, temporal, *result.x) - gain_energy(r['velocity'] - h * direction, r['speed'], r['dt'], p, temporal, *result.x)) / (2 * h)
    error = float(abs(numeric - (g * direction).sum()))
    fixed = velocity_energy(r['velocity'], r['speed'], r['dt'], p, temporal)
    nested = gain_energy(r['velocity'], r['speed'], r['dt'], p, temporal, 0.0, 0.0)
    if error > 3e-05 or abs(fixed - nested) > 1e-08:
        raise ValueError('Motion-scale derivative or exact fixed-scale nesting failed')
    prior.update(velocity_gain_marginalization=bool(selected), velocity_log_gain_prior_mean=float(result.x[0]), velocity_log_gain_prior_sigma=float(result.x[1]), velocity_gain_source_heldout_NLL=validation, velocity_fixed_gain_source_heldout_NLL=old, velocity_gain_source_train_NLL=float(result.fun), velocity_gain_source_gradient_difference=error, velocity_gain_source_training_43_selection_11=True, velocity_gain_target_labels_used=False, velocity_gain_integration_nodes=9, velocity_gain_fit_seconds=time.monotonic() - began)
    write(a.output, prior)
    print(json.dumps(dict(selected=selected, source_heldout_NLL=validation, old_fixed_gain_heldout_NLL=old, prior_median_gain=math.exp(result.x[0]), prior_log_sigma=float(result.x[1]), gradient_difference=error, fit_converged=bool(result.success), seconds=time.monotonic() - began)), flush=True)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'features', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    fit(p.parse_args())
if __name__ == '__main__':
    main()
