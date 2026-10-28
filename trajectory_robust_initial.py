import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.special import expit
from engine_utils import write
from trajectory_joint_calibration import field
from trajectory_joint_marginal import initial_prior, mixture_posterior

def population(priors):
    means = []
    seconds = []
    for p in priors:
        weight = np.exp(p['log_weight'])
        mu = p['mean']
        means.append((weight[:, None] * mu).sum(0))
        seconds.append((weight[:, None, None] * (p['covariance'] + mu[:, :, None] * mu[:, None, :])).sum(0))
    mean = np.mean(means, 0)
    cov = np.mean(seconds, 0) - np.outer(mean, mean)
    cov = 0.5 * (cov + cov.T) + np.eye(6) * 0.01
    return (mean, cov)

def augment(p, mean, covariance, weight, scale):
    if weight <= 0:
        return p
    mean = np.asarray(mean)
    covariance = np.asarray(covariance)
    means = np.concatenate((p['mean'], np.broadcast_to(mean, (2, 6))))
    cov = np.concatenate((p['covariance'], np.asarray([covariance, covariance * 9]) * scale ** 2))
    precision = np.linalg.inv(cov)
    log_weight = np.r_[p['log_weight'] + math.log1p(-weight), np.log([0.9, 0.1]) + math.log(weight)]
    return dict(mean=means, covariance=cov, precision=precision, logdet=np.linalg.slogdet(cov)[1], vector=np.einsum('kde,ke->kd', precision, means), log_weight=log_weight)

def robust_initial_prior(pre, configuration):
    p = initial_prior(pre, configuration.get('initial_device_covariance_scale', 1.0))
    weight = float(configuration.get('initial_population_mixture_weight', 0.0))
    return augment(p, configuration['initial_population_mean'], configuration['initial_population_covariance'], weight, configuration['initial_population_covariance_scale']) if weight > 0 else p

def fit(a):
    from trajectory_temporal_radio import ObservationModel
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    radio = field(a.base, prior)
    labels = {}
    rows = []
    for ordinal, s in enumerate(segments):
        name = s['device']
        if name not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target device in population-prior fit')
        if name not in labels:
            labels[name] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / name / 'evaluation_labels.json').read_text())
        folder = a.capture / 'source_train' / s['trajectory_id']
        with np.load(folder / 'PRE_forecasts.npz', allow_pickle=False) as z:
            pre = {k: z[k].copy() for k in z.files}
        saved = json.loads((folder / 'predictions.json').read_text())
        available = [p for p in saved if p['observation_id'] in labels[name]]
        t = np.r_[0.0, [(p['timestamp_ns'] - s['start_ns']) / 1000000000.0 for p in available]]
        xy = np.vstack((s['initial_position_xy_m'], [labels[name][p['observation_id']]['gt_xy_m'] for p in available]))
        model = ObservationModel(a.queries / 'source_train' / s['trajectory_id'] / 'radio_queries.npz', pre)
        model.mask &= np.asarray([str(oid) in labels[name] for oid in pre['observation_ids']])[:, None]
        if prior.get('radio_field_context_mode') == 'initial_five_final':
            model.freeze_field_context()
        if prior.get('geometry_locked_device_basis', False):
            model.lock_device_basis(s['initial_position_xy_m'], radio)
        raw_xy = np.stack([np.interp(model.raw_times.clip(0, t[-1]), t, xy[:, j]) for j in range(2)], -1)
        mean, B = model.geometry(raw_xy, radio)
        cov, _, _ = model.noise(np.asarray(prior['temporal_RSSI_parameters']))
        z, A, ld = model.whiten(mean, B, model.blocks(cov))
        rows.append(dict(z=z, A=A, ld=ld, prior=initial_prior(pre, prior.get('initial_device_covariance_scale', 1.0)), phone=name, heldout=ordinal % 5 == 0))
    training = [row for row in rows if not row['heldout']]
    populations = {name: population([row['prior'] for row in training if row['phone'] != name]) for name in labels}

    def objective(parameters, heldout):
        weight = expit(parameters[0])
        scale = math.exp(parameters[1])
        total = 0.0
        n = 0
        for row in rows:
            if row['heldout'] != heldout:
                continue
            m, P = populations[row['phone']]
            p = augment(row['prior'], m, P, weight, scale)
            total += mixture_posterior(row['z'], row['A'], p, row['ld'])[0]
            n += len(row['z'])
        return float(total / n)
    baseline = sum((mixture_posterior(r['z'], r['A'], r['prior'], r['ld'])[0] for r in rows if r['heldout'])) / sum((len(r['z']) for r in rows if r['heldout']))
    fits = []
    for initial in [[-2.0, 0.0], [0.0, math.log(2.0)]]:
        result = minimize(objective, initial, args=(False,), method='L-BFGS-B', bounds=[(-12.0, 4.0), (math.log(0.5), math.log(10.0))], options=dict(maxiter=60, ftol=1e-08))
        fits.append(dict(parameters=result.x.tolist(), source_train_NLL=float(result.fun), source_heldout_NLL=objective(result.x, True), converged=bool(result.success)))
    selected = min(fits, key=lambda r: r['source_heldout_NLL'])
    accepted = selected['source_heldout_NLL'] < baseline
    parameters = np.asarray(selected['parameters'])
    m, P = population([row['prior'] for row in training])
    prior.update(initial_population_mixture_weight=float(expit(parameters[0])) if accepted else 0.0, initial_population_covariance_scale=float(math.exp(parameters[1])), initial_population_mean=m.tolist(), initial_population_covariance=P.tolist(), initial_population_source_heldout_NLL=selected['source_heldout_NLL'], initial_population_original_source_heldout_NLL=baseline, initial_population_selected=bool(accepted), initial_population_fits=fits, initial_population_target_labels_used=False, initial_population_fit_phone_crossfit=True, initial_population_training_trajectories=len(training), initial_population_eight_initial_components_preserved=True, initial_population_fit_seconds=time.monotonic() - began)
    write(a.output, prior)
    print(json.dumps(dict(selected=accepted, weight=prior['initial_population_mixture_weight'], population_scale=prior['initial_population_covariance_scale'], source_heldout_NLL=selected['source_heldout_NLL'], original_heldout_NLL=baseline, source_phone_crossfit=True, seconds=time.monotonic() - began)), flush=True)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'queries', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    fit(p.parse_args())
if __name__ == '__main__':
    main()
