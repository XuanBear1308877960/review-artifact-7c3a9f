import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import block_diag, cholesky, cho_solve
from scipy.optimize import minimize
from scipy.special import expit
from engine_utils import write
from trajectory_joint_calibration import field
from trajectory_joint_marginal import initial_prior, mixture_posterior
from trajectory_robust_initial import population

class StartConditionalObservations:

    def whiten(self, mean, basis, blocks):
        return super().whiten(mean + self.start_offset, basis - self.start_basis_offset, blocks)

    def initial_noise(self, rows, ap, eta):
        return self.start_gain[rows, ap] * (self.start_residual[ap] - self.start_basis[ap] @ eta)

def conditioned_model(original, pre, known_xy, radio, covariances, shared, population_mean, population_covariance, sigma, velocity_sigma):
    from trajectory_temporal_radio import ObservationModel

    class Conditional(StartConditionalObservations, ObservationModel):
        pass
    model = object.__new__(Conditional)
    model.__dict__.update(original.__dict__)
    m, B, dm, db = radio(np.asarray(known_xy, float)[None], pre['motion'][:1], True)
    m, B, dm, db = (v[0] for v in (m, B, dm, db))
    mu = np.asarray(population_mean)
    P = np.asarray(population_covariance)
    gradient = dm + np.einsum('adz,d->az', db, mu)
    gradient_second = (gradient * gradient).sum(-1) + np.einsum('adz,de,aez->a', db, P, db)
    age = np.zeros(4)
    valid = pre['mask'][0].copy()
    for ap in range(4):
        index = np.where(model.S[ap][0] > 0)[0]
        times = np.asarray(model.query_metadata['raw_times_s_' + str(ap)])[index]
        age[ap] = max(abs(times), default=float('inf'))
        valid[ap] &= bool(len(index)) and age[ap] <= 0.5
    variance = np.array([c[0, 0] for c in covariances]) + sigma ** 2
    variance += velocity_sigma ** 2 * age.clip(0, 0.5) ** 2 * gradient_second
    start_residual = pre['rssi'][0] - m
    precision = np.linalg.inv(P)
    vector = precision @ mu
    if valid.any():
        precision += B[valid].T @ (B[valid] / variance[valid, None])
        vector += B[valid].T @ (start_residual[valid] / variance[valid])
    L = cholesky(precision, lower=True, check_finite=False)
    postcov = cho_solve((L, True), np.eye(6), check_finite=False)
    postmean = postcov @ vector
    gain = np.zeros((model.N, 4))
    conditional = []
    conditional_shared = []
    for ap in range(4):
        cross = covariances[ap][:, 0].copy() if valid[ap] else np.zeros(model.N)
        gain[:, ap] = cross / variance[ap]
        removed = np.outer(cross, cross) / variance[ap]
        conditional.append(covariances[ap] - removed)
        conditional_shared.append(shared[ap] - removed)
    model.start_gain = gain
    model.start_residual = start_residual
    model.start_basis = B
    model.start_variance = variance
    model.start_valid = valid
    model.start_offset = gain * start_residual[None]
    model.start_basis_offset = gain[..., None] * B[None]
    diagnostic = dict(first_scan_valid_APs=valid.tolist(), first_scan_max_contributor_age_s=[float(v) if np.isfinite(v) else None for v in age], first_scan_additional_noise_db=float(sigma), first_scan_motion_sigma_m_s=float(velocity_sigma), first_scan_covariance_conditioned=True, known_start_only=True, initial_radio_drift_approximation='Per-AP isotropic constant-velocity variance; source-fitted speed second moment')
    return (model, postmean, postcov, conditional, conditional_shared, diagnostic)

def conditional_gaussian_proof(row, radio, mu, P, sigma, velocity_sigma):
    model, m, C, cov, _, _ = conditioned_model(row['model'], row['pre'], row['start'], radio, row['cov'], row['shared'], mu, P, sigma, velocity_sigma)
    valid = np.where(model.start_valid)[0]
    r0 = model.start_residual[valid]
    B0 = model.start_basis[valid]
    rs = []
    Bs = []
    noise = []
    cross = []
    for ap, ix, _ in row['model'].blocks(row['cov']):
        rs.append(row['pre']['rssi'][ix, ap] - row['mean'][ix, ap])
        Bs.append(row['basis'][ix, ap])
        noise.append(row['cov'][ap][np.ix_(ix, ix)])
        X = np.zeros((len(ix), len(valid)))
        if ap in valid:
            X[:, int(np.where(valid == ap)[0][0])] = row['cov'][ap][ix, 0]
        cross.append(X)
    r = np.concatenate(rs)
    B = np.concatenate(Bs)
    X = np.concatenate(cross)
    R0 = np.diag(model.start_variance[valid])
    joint_noise = np.block([[R0, X.T], [X, block_diag(*noise)]])
    BJ = np.vstack((B0, B))
    rJ = np.r_[r0, r]

    def nll(y, design, R):
        L = cholesky(R + design @ P @ design.T, lower=True, check_finite=False)
        e = y - design @ mu
        return 0.5 * (e @ cho_solve((L, True), e, check_finite=False) + 2 * np.log(np.diag(L)).sum() + len(e) * math.log(2 * math.pi))
    direct = nll(rJ, BJ, joint_noise) - nll(r0, B0, R0)
    z, A, ld = model.whiten(row['mean'], row['basis'], model.blocks(cov))
    one = dict(mean=m[None], covariance=C[None], precision=np.linalg.inv(C)[None], vector=np.linalg.solve(C, m)[None], logdet=np.array([np.linalg.slogdet(C)[1]]), log_weight=np.zeros(1))
    actual = mixture_posterior(z, A, one, ld)[0]
    error = float(abs(direct - actual))
    if error > 1e-06:
        raise ValueError('Known-start Schur likelihood differs from full dense Gaussian')
    return error

def mix(p, mean, covariance, weight):
    means = np.concatenate((p['mean'], np.asarray(mean)[None]))
    cov = np.concatenate((p['covariance'], np.asarray(covariance)[None]))
    precision = np.linalg.inv(cov)
    return dict(mean=means, covariance=cov, precision=precision, vector=np.einsum('kde,ke->kd', precision, means), logdet=np.linalg.slogdet(cov)[1], log_weight=np.r_[p['log_weight'] + math.log1p(-weight), math.log(weight)])

def configure(pre, configuration, known_xy, radio, observations, covariances, shared):
    model, mu, P, cov, shr, diagnostic = conditioned_model(observations, pre, known_xy, radio, covariances, shared, configuration['known_start_population_mean'], configuration['known_start_population_covariance'], configuration['known_start_extra_radio_noise_db'], configuration['known_start_velocity_sigma_m_s'])
    p = mix(initial_prior(pre, configuration.get('initial_device_covariance_scale', 1.0)), mu, P, configuration['known_start_population_mixture_weight'])
    return (model, p, cov, shr, diagnostic)

def fit(a):
    from trajectory_temporal_radio import ObservationModel
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    radio = field(a.base, prior)
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    labels = {}
    rows = []
    velocity = []
    for ordinal, s in enumerate(segments):
        phone = s['device']
        if phone not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target phone in known-start calibration training')
        if phone not in labels:
            labels[phone] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / phone / 'evaluation_labels.json').read_text())
        folder = a.capture / 'source_train' / s['trajectory_id']
        with np.load(folder / 'PRE_forecasts.npz', allow_pickle=False) as z:
            pre = {k: z[k].copy() for k in z.files}
        saved = json.loads((folder / 'predictions.json').read_text())
        available = [r for r in saved if r['observation_id'] in labels[phone]]
        t = np.r_[0.0, [(r['timestamp_ns'] - s['start_ns']) / 1000000000.0 for r in available]]
        xy = np.vstack((s['initial_position_xy_m'], [labels[phone][r['observation_id']]['gt_xy_m'] for r in available]))
        if ordinal % 5:
            grid = np.arange(0, max(0, t[-1] - 1), 1.0)
            displacement = np.stack([np.interp(grid + 1, t, xy[:, j]) - np.interp(grid, t, xy[:, j]) for j in range(2)], -1)
            velocity.extend(np.square(displacement).sum(-1).tolist())
        model = ObservationModel(a.queries / 'source_train' / s['trajectory_id'] / 'radio_queries.npz', pre)
        model.mask &= np.asarray([str(oid) in labels[phone] for oid in pre['observation_ids']])[:, None]
        raw_xy = np.stack([np.interp(model.raw_times.clip(0, t[-1]), t, xy[:, j]) for j in range(2)], -1)
        mean, B = model.geometry(raw_xy, radio)
        cov, shared, _ = model.noise(np.asarray(prior['temporal_RSSI_parameters']))
        p = initial_prior(pre, prior.get('initial_device_covariance_scale', 1.0))
        rows.append(dict(pre=pre, model=model, mean=mean, basis=B, cov=cov, shared=shared, prior=p, phone=phone, heldout=ordinal % 5 == 0, start=s['initial_position_xy_m']))
    training = [r for r in rows if not r['heldout']]
    populations = {phone: population([r['prior'] for r in training if r['phone'] != phone]) for phone in labels}
    velocity_sigma = math.sqrt(float(np.mean(velocity)) / 2)
    cache = {}

    def values(logsigma, heldout):
        key = (float(logsigma), heldout)
        if key in cache:
            return cache[key]
        old = []
        new = []
        weights = []
        count = 0
        for r in rows:
            if r['heldout'] != heldout:
                continue
            mu, P = populations[r['phone']]
            model, m, C, covs, _, _ = conditioned_model(r['model'], r['pre'], r['start'], radio, r['cov'], r['shared'], mu, P, math.exp(logsigma), velocity_sigma)
            z, A, ld = model.whiten(r['mean'], r['basis'], model.blocks(covs))
            old.append(mixture_posterior(z, A, r['prior'], ld)[0])
            one = mix(r['prior'], m, C, 0.5)
            new.append(mixture_posterior(z, A, {k: v[-1:] for k, v in one.items()}, ld)[0] - math.log(2))
            count += len(z)
        result = (np.asarray(old), np.asarray(new), count)
        cache[key] = result
        return result

    def objective(parameters, heldout):
        weight = expit(parameters[0])
        old, new, count = values(parameters[1], heldout)
        return float((-np.logaddexp(math.log1p(-weight) - old, math.log(weight) - new)).sum() / count)
    baseline = 0.0
    n = 0
    for r in rows:
        if not r['heldout']:
            continue
        z, A, ld = r['model'].whiten(r['mean'], r['basis'], r['model'].blocks(r['cov']))
        baseline += mixture_posterior(z, A, r['prior'], ld)[0]
        n += len(z)
    baseline /= n
    fits = []
    for start in [[-2.0, math.log(2.0)], [0.0, math.log(6.0)]]:
        result = minimize(objective, start, args=(False,), method='L-BFGS-B', bounds=[(-10, 5), (math.log(0.5), math.log(20))], options=dict(maxiter=50, ftol=1e-08))
        fits.append(dict(parameters=result.x.tolist(), source_train_NLL=float(result.fun), source_heldout_NLL=objective(result.x, True), converged=bool(result.success)))
    selected = min(fits, key=lambda f: f['source_heldout_NLL'])
    weight = expit(selected['parameters'][0])
    sigma = math.exp(selected['parameters'][1])
    sample = training[0]
    cross_mu, cross_P = populations[sample['phone']]
    proof = conditional_gaussian_proof(sample, radio, cross_mu, cross_P, sigma, velocity_sigma)
    mu, P = population([r['prior'] for r in training])
    prior.update(known_start_radio_conditioning=True, known_start_population_mean=mu.tolist(), known_start_population_covariance=P.tolist(), known_start_population_mixture_weight=float(weight), known_start_extra_radio_noise_db=sigma, known_start_velocity_sigma_m_s=velocity_sigma, known_start_source_train_trajectories=43, known_start_source_heldout_trajectories=11, known_start_source_phone_crossfit=True, known_start_original_eight_components_retained=True, known_start_recognition_and_first_scan_mixture_not_product=True, known_start_later_positions_used_for_inference=False, known_start_target_labels_used=False, known_start_source_fits=fits, known_start_source_heldout_NLL=selected['source_heldout_NLL'], known_start_original_source_heldout_NLL=baseline, known_start_source_likelihood_improves=selected['source_heldout_NLL'] < baseline, known_start_dense_Gaussian_likelihood_difference=proof, known_start_fit_seconds=time.monotonic() - began)
    write(a.output, prior)
    print(json.dumps(dict(weight=float(weight), extra_sigma_db=sigma, velocity_sigma_m_s=velocity_sigma, source_heldout_NLL=selected['source_heldout_NLL'], original_source_heldout_NLL=baseline, seconds=time.monotonic() - began)), flush=True)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'queries', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    fit(p.parse_args())
if __name__ == '__main__':
    main()
