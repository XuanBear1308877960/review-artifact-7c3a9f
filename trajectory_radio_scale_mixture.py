import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from engine_utils import write
from trajectory_joint_calibration import field
from trajectory_joint_marginal import initial_prior, mixture_posterior

def scale_posterior(z, A, prior, logdet, configuration, derivative=False):
    scales = np.asarray(configuration['RSSI_segment_scale_factors'], float)
    logprior = np.log(configuration['RSSI_segment_scale_probability'])
    values = [mixture_posterior(z / s, A / s, prior, logdet + 2 * len(z) * math.log(s)) for s in scales]
    scores = logprior - np.asarray([v[0] for v in values])
    normalizer = logsumexp(scores)
    weight = np.exp(scores - normalizer)
    means = np.asarray([v[1] for v in values])
    mean = weight @ means
    difference = means - mean
    covariance = np.einsum('k,kde->de', weight, np.asarray([v[2] for v in values]) + difference[:, :, None] * difference[:, None, :])
    components = weight @ np.asarray([v[4] for v in values])
    result = (-float(normalizer), mean, covariance, z - A @ mean, components)
    if not derivative:
        return result
    gz = np.zeros_like(z)
    gA = np.zeros_like(A)
    for w, s, v in zip(weight, scales, values):
        residual = z - A @ v[1]
        gz += w / s ** 2 * residual
        gA += w / s ** 2 * (A @ v[2] - residual[:, None] * v[1][None])
    return result + (dict(g_z=gz, g_A=gA, scale_posterior=weight),)

def fit(a):
    from trajectory_temporal_radio import ObservationModel
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    radio = field(a.base, prior)
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    labels = {}
    rows = []
    scales = np.asarray([0.5, 1.0, 2.0])
    configuration = dict(RSSI_segment_scale_factors=scales.tolist(), RSSI_segment_scale_probability=[0.1, 0.8, 0.1])
    for ordinal, s in enumerate(segments):
        phone = s['device']
        if phone not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target reference in segment-noise fit')
        if phone not in labels:
            labels[phone] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / phone / 'evaluation_labels.json').read_text())
        folder = a.capture / 'source_train' / s['trajectory_id']
        with np.load(folder / 'PRE_forecasts.npz', allow_pickle=False) as z:
            pre = {k: z[k].copy() for k in z.files}
        saved = json.loads((folder / 'predictions.json').read_text())
        available = [r for r in saved if r['observation_id'] in labels[phone]]
        t = np.r_[0.0, [(r['timestamp_ns'] - s['start_ns']) / 1000000000.0 for r in available]]
        xy = np.vstack((s['initial_position_xy_m'], [labels[phone][r['observation_id']]['gt_xy_m'] for r in available]))
        model = ObservationModel(a.queries / 'source_train' / s['trajectory_id'] / 'radio_queries.npz', pre)
        model.mask &= np.asarray([str(oid) in labels[phone] for oid in pre['observation_ids']])[:, None]
        p = initial_prior(pre, prior.get('initial_device_covariance_scale', 1.0))
        cov, shared, _ = model.noise(np.asarray(prior['temporal_RSSI_parameters']))
        if prior.get('known_start_radio_conditioning', False):
            from trajectory_known_start_radio import configure
            model, p, cov, shared, _ = configure(pre, prior, s['initial_position_xy_m'], radio, model, cov, shared)
        raw_xy = np.stack([np.interp(model.raw_times.clip(0, t[-1]), t, xy[:, j]) for j in range(2)], -1)
        mean, B = model.geometry(raw_xy, radio)
        z, A, ld = model.whiten(mean, B, model.blocks(cov))
        values = [mixture_posterior(z / sigma, A / sigma, p, ld + 2 * len(z) * math.log(sigma))[0] for sigma in scales]
        rows.append(dict(z=z, A=A, ld=ld, prior=p, n=len(z), heldout=ordinal % 5 == 0, values=values))
    E = np.asarray([r['values'] for r in rows])
    held = np.asarray([r['heldout'] for r in rows])
    n = np.asarray([r['n'] for r in rows])

    def objective(logits, heldout, derivative=False):
        logpi = logits - logsumexp(logits)
        posterior = softmax(logpi - E[held == heldout], axis=-1)
        loss = -float(logsumexp(logpi - E[held == heldout], axis=-1).sum())
        grad = (softmax(logits)[None] - posterior).sum(0)
        if not heldout:
            loss -= 0.5 * logpi.sum()
            grad += 0.5 * (3 * softmax(logits) - 1)
        denominator = float(n[held == heldout].sum())
        return (loss / denominator, grad / denominator) if derivative else loss / denominator
    fitted = minimize(objective, np.log([0.1, 0.8, 0.1]), args=(False, True), jac=True, method='L-BFGS-B', options=dict(maxiter=100, ftol=1e-12))
    pi = softmax(fitted.x)
    configuration['RSSI_segment_scale_probability'] = pi.tolist()
    r = rows[0]
    rng = np.random.RandomState(82619)
    dz = rng.normal(size=r['z'].shape)
    dA = rng.normal(size=r['A'].shape)
    norm = math.sqrt(float(dz @ dz + (dA * dA).sum()))
    dz /= norm
    dA /= norm
    step = 1e-05
    actual = scale_posterior(r['z'], r['A'], r['prior'], r['ld'], configuration, True)
    numeric = (scale_posterior(r['z'] + step * dz, r['A'] + step * dA, r['prior'], r['ld'], configuration)[0] - scale_posterior(r['z'] - step * dz, r['A'] - step * dA, r['prior'], r['ld'], configuration)[0]) / (2 * step)
    gradient = actual[-1]
    difference = float(abs(numeric - gradient['g_z'] @ dz - (gradient['g_A'] * dA).sum()))
    if difference > 2e-05:
        raise ValueError('Segment-scale marginalized RSSI gradient differs')
    baseline = float(E[held, 1].sum() / n[held].sum())
    val = objective(fitted.x, True)
    posterior = softmax(np.log(pi) - E[held], axis=-1)
    prior.update(configuration)
    prior.update(RSSI_segment_scale_marginalization=True, RSSI_segment_scale_source_train_trajectories=43, RSSI_segment_scale_source_heldout_trajectories=11, RSSI_segment_scale_source_heldout_NLL=val, RSSI_segment_scale_fixed_source_heldout_NLL=baseline, RSSI_segment_scale_source_heldout_posterior_mean=posterior.mean(0).tolist(), RSSI_segment_scale_source_heldout_high_noise_fraction=float((posterior[:, -1] > 0.5).mean()), RSSI_segment_scale_original_initial_device_priors_unchanged=True, RSSI_segment_scale_target_labels_used=False, RSSI_segment_scale_gradient_difference=difference, RSSI_segment_scale_fit_seconds=time.monotonic() - began, cache_solver_groups=True, save_solver_state=True)
    write(a.output, prior)
    print(json.dumps(dict(scale_factors=scales.tolist(), source_probabilities=pi.tolist(), source_heldout_NLL=val, fixed_noise_NLL=baseline, high_noise_fraction=prior['RSSI_segment_scale_source_heldout_high_noise_fraction'], seconds=time.monotonic() - began)), flush=True)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'queries', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    fit(p.parse_args())
if __name__ == '__main__':
    main()
