import argparse
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from engine_utils import write
from trajectory_joint_marginal import mixture_posterior
STATES = np.asarray(list(itertools.product([0, 1], repeat=4)), float)

def family_parameters(configuration):
    probability = np.asarray(configuration['RSSI_AP_high_noise_probability'], float)
    if probability.shape != (4,) or np.any((probability <= 0) | (probability >= 1)):
        raise ValueError('Per-AP probabilities must be strictly between zero and one')
    scale = float(configuration['RSSI_AP_high_noise_scale'])
    if not scale > 1:
        raise ValueError('The broad Gaussian family must have scale > 1')
    logprior = (STATES * np.log(probability) + (1 - STATES) * np.log1p(-probability)).sum(-1)
    inverse_variance = np.exp(-2 * math.log(scale) * STATES)
    return (logprior, inverse_variance)

def ap_posterior(z, A, prior, logdet, blocks, configuration, derivative=False):
    logprior, inverse_variance = family_parameters(configuration)
    AA = np.zeros((4, 6, 6))
    Az = np.zeros((4, 6))
    zz = np.zeros(4)
    counts = np.zeros(4)
    offset = 0
    row_ap = np.zeros(len(z), int)
    for ap, ix, _ in blocks:
        n = len(ix)
        sl = slice(offset, offset + n)
        offset += n
        row_ap[sl] = ap
        AA[ap] = A[sl].T @ A[sl]
        Az[ap] = A[sl].T @ z[sl]
        zz[ap] = z[sl] @ z[sl]
        counts[ap] = n
    if offset != len(z):
        raise ValueError('Per-AP Gaussian block lengths differ')
    precision = prior['precision'][None] + np.einsum('ba,aij->bij', inverse_variance, AA)[:, None]
    vector = prior['vector'][None] + np.einsum('ba,ai->bi', inverse_variance, Az)[:, None]
    L = np.linalg.cholesky(precision)
    covariance = np.linalg.solve(np.swapaxes(L, -1, -2), np.linalg.solve(L, np.broadcast_to(np.eye(6), precision.shape)))
    means = np.einsum('bkij,bkj->bki', covariance, vector)
    extra_logdet = 2 * math.log(configuration['RSSI_AP_high_noise_scale']) * (STATES @ counts)
    prior_quadratic = np.einsum('ki,kij,kj->k', prior['mean'], prior['precision'], prior['mean'])
    values = 0.5 * (np.einsum('ba,a->b', inverse_variance, zz)[:, None] - np.einsum('bki,bki->bk', vector, means) + prior_quadratic[None] + logdet + extra_logdet[:, None] + prior['logdet'][None] + 2 * np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(-1) + len(z) * math.log(2 * math.pi))
    score = logprior[:, None] + prior['log_weight'][None] - values
    normalizer = logsumexp(score)
    joint = np.exp(score - normalizer)
    mean = np.einsum('bk,bki->i', joint, means)
    difference = means - mean
    P = np.einsum('bk,bkij->ij', joint, covariance + difference[..., None] * difference[..., None, :])
    device_weight = joint.sum(0)
    family_weight = joint.sum(1)
    result = (-float(normalizer), mean, P, z - A @ mean, device_weight)
    if not derivative:
        return result
    gz = np.zeros_like(z)
    gA = np.zeros_like(A)
    second = covariance + means[..., None] * means[..., None, :]
    for ap in range(4):
        ix = np.where(row_ap == ap)[0]
        if not len(ix):
            continue
        mass = joint * inverse_variance[:, ap, None]
        first = np.einsum('bk,bki->i', mass, means)
        moment = np.einsum('bk,bkij->ij', mass, second)
        gz[ix] = z[ix] * mass.sum() - A[ix] @ first
        gA[ix] = A[ix] @ moment - z[ix, None] * first[None]
    return result + (dict(g_z=gz, g_A=gA, family_posterior=family_weight, AP_high_noise_posterior=family_weight @ STATES, family_values=-logsumexp(prior['log_weight'][None] - values, axis=-1)),)

def fit(a):
    from trajectory_spatial_residual import source_rows
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    if any((prior.get(k) for k in ['RSSI_segment_scale_marginalization', 'known_start_radio_conditioning', 'geometry_locked_device_basis', 'field_context_override_root'])) or prior.get('device_spatial_basis_strength', 1.0) != 1.0:
        raise ValueError('Evaluate AP families independently before combining initial-noise mechanisms')
    rows, _ = source_rows(a, prior)
    for row in rows:
        model = row['model']
        cov, _, _ = model.noise(np.asarray(prior['temporal_RSSI_parameters']))
        row['blocks'] = model.blocks(cov)
        row['z'], row['A'], row['ld'] = model.whiten(row['mean'], row['basis'], row['blocks'])
    held = np.asarray([r['heldout'] for r in rows])
    counts = np.asarray([len(r['z']) for r in rows])
    baseline = float(sum((mixture_posterior(r['z'], r['A'], r['prior'], r['ld'])[0] for r in rows if r['heldout'])) / counts[held].sum())
    trials = []
    for scale in [2.0, 4.0]:
        config = dict(RSSI_AP_high_noise_probability=[0.1] * 4, RSSI_AP_high_noise_scale=scale)
        E = np.asarray([ap_posterior(r['z'], r['A'], r['prior'], r['ld'], r['blocks'], config, True)[-1]['family_values'] for r in rows])

        def objective(logits, heldout, gradient=False):
            p = expit(logits)
            logpi = (STATES * np.log(p) + (1 - STATES) * np.log1p(-p)).sum(-1)
            scores = logpi - E[held == heldout]
            weight = np.exp(scores - logsumexp(scores, axis=-1)[:, None])
            value = -float(logsumexp(scores, axis=-1).sum())
            g = (p[None] - weight @ STATES).sum(0)
            if not heldout:
                value -= 0.5 * float((np.log(p) + np.log1p(-p)).sum())
                g += p - 0.5
            denominator = float(counts[held == heldout].sum())
            return (value / denominator, g / denominator) if gradient else value / denominator
        result = minimize(objective, np.full(4, math.log(0.1 / 0.9)), args=(False, True), jac=True, method='L-BFGS-B', bounds=[(-8.0, 8.0)] * 4, options=dict(maxiter=100, ftol=1e-12))
        config['RSSI_AP_high_noise_probability'] = expit(result.x).tolist()
        posterior = np.asarray([ap_posterior(r['z'], r['A'], r['prior'], r['ld'], r['blocks'], config, True)[-1]['AP_high_noise_posterior'] for r in rows if r['heldout']])
        trials.append(dict(configuration=config, source_heldout_NLL=objective(result.x, True), source_train_NLL=float(result.fun), source_heldout_AP_high_noise_posterior_mean=posterior.mean(0).tolist(), source_heldout_AP_high_noise_fraction=(posterior > 0.5).mean(0).tolist(), converged=bool(result.success)))
    chosen = min(trials, key=lambda r: r['source_heldout_NLL'])
    config = chosen['configuration']
    r = rows[0]
    lp, iv = family_parameters(config)
    scalar = []
    row_ap = np.concatenate([np.full(len(ix), ap) for ap, ix, _ in r['blocks']])
    for b in range(len(STATES)):
        scale = 1 / np.sqrt(iv[b, row_ap])
        extra = 2 * np.log(scale).sum()
        scalar.append(mixture_posterior(r['z'] / scale, r['A'] / scale[:, None], r['prior'], r['ld'] + extra)[0])
    reference = -float(logsumexp(lp - np.asarray(scalar)))
    actual = ap_posterior(r['z'], r['A'], r['prior'], r['ld'], r['blocks'], config, True)
    reproduction = float(abs(actual[0] - reference))
    if reproduction > 1e-07:
        raise ValueError('Batched AP mixture differs from scalar Gaussian evidence')
    rng = np.random.RandomState(83161)
    dz = rng.normal(size=r['z'].shape)
    dA = rng.normal(size=r['A'].shape)
    norm = math.sqrt(float(dz @ dz + (dA * dA).sum()))
    dz /= norm
    dA /= norm
    h = 1e-05
    numeric = (ap_posterior(r['z'] + h * dz, r['A'] + h * dA, r['prior'], r['ld'], r['blocks'], config)[0] - ap_posterior(r['z'] - h * dz, r['A'] - h * dA, r['prior'], r['ld'], r['blocks'], config)[0]) / (2 * h)
    difference = float(abs(numeric - actual[-1]['g_z'] @ dz - (actual[-1]['g_A'] * dA).sum()))
    if difference > 2e-05:
        raise ValueError('Per-AP mixture marginal gradient differs')
    prior.update(config)
    prior.update(RSSI_AP_scale_marginalization=True, RSSI_AP_scale_source_trials=trials, RSSI_AP_scale_source_heldout_NLL=chosen['source_heldout_NLL'], RSSI_AP_scale_baseline_heldout_NLL=baseline, RSSI_AP_scale_source_train_trajectories=43, RSSI_AP_scale_source_heldout_trajectories=11, RSSI_AP_scale_initial_eight_unchanged=True, RSSI_AP_scale_target_labels_used=False, RSSI_AP_scale_normalizers_retained=True, RSSI_AP_scale_scalar_reproduction_difference=reproduction, RSSI_AP_scale_gradient_difference=difference, RSSI_AP_scale_fit_seconds=time.monotonic() - began, cache_solver_groups=True, save_solver_state=True)
    write(a.output, prior)
    print(json.dumps(dict(configuration=config, source_heldout_NLL=chosen['source_heldout_NLL'], baseline_NLL=baseline, scalar_difference=reproduction, gradient_difference=difference, seconds=time.monotonic() - began)), flush=True)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'queries', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    fit(p.parse_args())
if __name__ == '__main__':
    main()
