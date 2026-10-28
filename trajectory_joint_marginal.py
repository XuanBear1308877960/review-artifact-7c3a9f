import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular, cho_solve
from scipy.optimize import minimize
from scipy.special import logsumexp
from engine_utils import write, score_cohort
from trajectory_joint_calibration import field, source_rows, radio_blocks
from trajectory_raw_hmm import ForceRawHMMPath
from trajectory_score import shape
from temporal_radio import PARENT_RELATIVE

def initial_prior(pre, uncertainty_scale=1.0):
    if len(pre['device_mean']) <= 5:
        raise ValueError('Missing complete initial-five posterior')
    mean = pre['device_mean'][5].astype(float)
    cov = pre['device_covariance'][5].astype(float)
    cov = 0.5 * (cov + cov.swapaxes(-1, -2)) + np.eye(6)[None] * 0.0001
    cov *= float(uncertainty_scale) ** 2
    precision = np.linalg.inv(cov)
    logdet = np.linalg.slogdet(cov)[1]
    weight = pre['weights'][5].astype(float)
    weight /= weight.sum()
    return dict(mean=mean, covariance=cov, precision=precision, vector=np.einsum('kde,ke->kd', precision, mean), logdet=logdet, log_weight=np.log(weight.clip(1e-20)))

def whiten(t, mask, y, mean, basis, parameters):
    blocks = radio_blocks(t, mask, parameters)
    z = []
    A = []
    logdet = 0.0
    for ap, ix, L in blocks:
        z.append(solve_triangular(L, y[ix, ap] - mean[ix, ap], lower=True, check_finite=False))
        A.append(solve_triangular(L, basis[ix, ap], lower=True, check_finite=False))
        logdet += 2 * np.log(np.diag(L)).sum()
    if not z:
        return (np.zeros(0), np.zeros((0, 6)), 0.0, blocks)
    return (np.concatenate(z), np.concatenate(A), float(logdet), blocks)

def posterior(z, A, prior, mode, noise_logdet=0.0):
    precision = prior['precision'][mode] + A.T @ A
    L = cholesky(precision, lower=True, check_finite=False)
    covariance = cho_solve((L, True), np.eye(A.shape[1]), check_finite=False)
    mean = covariance @ (prior['vector'][mode] + A.T @ z)
    residual = z - A @ mean
    d = mean - prior['mean'][mode]
    value = 0.5 * float(residual @ residual + d @ prior['precision'][mode] @ d + noise_logdet + prior['logdet'][mode] + 2 * np.log(np.diag(L)).sum() + len(z) * math.log(2 * math.pi))
    return (value, mean, covariance, residual)

def mixture_posterior(z, A, prior, noise_logdet=0.0):
    values = [posterior(z, A, prior, k, noise_logdet) for k in range(len(prior['mean']))]
    evidence = prior['log_weight'] - np.asarray([v[0] for v in values])
    normalizer = logsumexp(evidence)
    weight = np.exp(evidence - normalizer)
    means = np.asarray([v[1] for v in values])
    covariances = np.asarray([v[2] for v in values])
    mean = (weight[:, None] * means).sum(0)
    difference = means - mean
    covariance = (weight[:, None, None] * (covariances + difference[:, :, None] * difference[:, None, :])).sum(0)
    return (-float(normalizer), mean, covariance, z - A @ mean, weight)

def fit(a):
    began = time.monotonic()
    rows = source_rows(a)
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    for r, s in zip(rows, segments):
        with np.load(a.capture / 'source_train' / s['trajectory_id'] / 'PRE_forecasts.npz', allow_pickle=False) as z:
            pre = {k: z[k].copy() for k in ['device_mean', 'device_covariance', 'weights']}
        r['initial'] = initial_prior(pre)

    def objective(parameters, heldout):
        total = 0.0
        count = 0
        for r in rows:
            if r['heldout'] != heldout:
                continue
            z, A, ld, _ = whiten(r['t'], r['mask'], r['rssi'], r['mean'], r['basis'], parameters)
            nll = np.array([posterior(z, A, r['initial'], mode, ld)[0] for mode in range(8)])
            total -= logsumexp(r['initial']['log_weight'] - nll)
            count += len(z)
        return float(total / count)
    initial = np.log([1.5] * 4 + [2.0] * 4 + [2.0])
    bounds = [(math.log(0.2), math.log(12.0))] * 4 + [(math.log(0.02), math.log(15.0))] * 4 + [(math.log(0.1), math.log(30.0))]
    result = minimize(objective, initial, args=(False,), method='L-BFGS-B', bounds=bounds, options=dict(maxiter=100, ftol=2e-07))
    prior = json.loads(a.prior.read_text())
    prior.update(joint_marginal_RSSI=True, joint_marginal_initial_mixture=True, save_all_calibration_candidates=False, joint_marginal_RSSI_parameters=result.x.tolist(), joint_runtime_base=str(a.base.resolve()), joint_marginal_source_train_NLL=float(result.fun), joint_marginal_source_heldout_NLL=objective(result.x, True), joint_marginal_source_initial_noise_heldout_NLL=objective(initial, True), joint_marginal_fit_converged=bool(result.success), joint_marginal_fit_termination=str(result.message), device_prior_PRE_index=5, immutable_initial_five_preserved=True, initial_five_RSSI_not_recounted=True, all_eight_initial_calibration_posteriors_retained=True, calibration_marginal_logdet_and_derivative_retained=True, source_map_IDW_radius_m=0.15, old_frozen_short_residual_mean_used=False, source_only_RSSI_noise_fit=True, joint_marginal_target_GT_used=False, final_RSSI_forecast_scope='Offline current-estimated-region observation-factor omission, conditional on immutable initial-five support, complete sensor-predicted path and frozen causal contexts', region_omission_scope='Rolling observation factors only', full_region_withheld_path_and_context_replay=False, source_fit_seconds=time.monotonic() - began)
    write(a.output, prior)
    print(json.dumps(dict(source_heldout_NLL=prior['joint_marginal_source_heldout_NLL'], initial_noise_heldout_NLL=prior['joint_marginal_source_initial_noise_heldout_NLL'], white_sigma_db=np.exp(result.x[:4]).tolist(), colored_sigma_db=np.exp(result.x[4:8]).tolist(), propagation_tau_s=float(np.exp(result.x[8])), seconds=time.monotonic() - began)), flush=True)

class JointMarginalPath(ForceRawHMMPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wifi_weight[:] = 0.0
        with np.load(args[0] / 'PRE_forecasts.npz', allow_pickle=False) as z:
            self.pre = {k: z[k].copy() for k in z.files}
        self.device_prior = initial_prior(self.pre)
        self.radio_field = field(Path(self.prior['joint_runtime_base']))
        self.radio_parameters = np.asarray(self.prior['joint_marginal_RSSI_parameters'])
        self.radio_blocks = radio_blocks(self.t, self.pre['mask'], self.radio_parameters)
        self.radio_noise_logdet = sum((float(2 * np.log(np.diag(L)).sum()) for _, _, L in self.radio_blocks))

    def radio(self, q, mode):
        xy = self.B @ q
        mean, basis, dm, db = self.radio_field(xy, self.pre['motion'], True)
        zs = []
        As = []
        for ap, ix, L in self.radio_blocks:
            zs.append(solve_triangular(L, self.pre['rssi'][ix, ap] - mean[ix, ap], lower=True, check_finite=False))
            As.append(solve_triangular(L, basis[ix, ap], lower=True, check_finite=False))
        z = np.concatenate(zs)
        A = np.concatenate(As)
        loss, eta, P, residual, component_weight = mixture_posterior(z, A, self.device_prior, self.radio_noise_logdet)
        gradient = np.zeros_like(xy)
        offset = 0
        for ap, ix, L in self.radio_blocks:
            end = offset + len(ix)
            score = solve_triangular(L.T, residual[offset:end], lower=False, check_finite=False)
            uncertainty = solve_triangular(L.T, A[offset:end] @ P, lower=False, check_finite=False)
            gradient[ix] -= score[:, None] * dm[ix, ap]
            gradient[ix] += np.einsum('nd,ndz->nz', uncertainty - score[:, None] * eta[None], db[ix, ap])
            offset = end
        return (float(loss), self.B.T @ gradient, (eta, P, component_weight))

    def path_objective(self, vector, mode):
        value, gradient = super().path_objective(vector, mode)
        q, _ = self.unpack(vector)
        loss, g, _ = self.radio(q, mode)
        value += loss
        gradient[:2 * (self.n - 1)] += g[1:].ravel()
        return (value, gradient)

    def solve_axis(self, alternative, modes):
        predictions, diagnostic = super().solve_axis(alternative, modes)
        q, _ = self.unpack(self.last_selected_vector[:-1])
        mode = diagnostic['calibration_mode']
        _, _, (eta, P, component_weight) = self.radio(q, mode)
        xy = self.B @ q
        mean, basis = self.radio_field(xy, self.pre['motion'])
        cells = np.floor(xy / 2.0).astype(int)
        keys = [tuple(c) for c in cells]
        regions = sorted(set(keys))
        N = len(xy)
        _, initial_mean, initial_cov, _, _ = mixture_posterior(np.zeros(0), np.zeros((0, 6)), self.device_prior)
        means = np.broadcast_to(initial_mean, (N, 6)).copy()
        covariances = np.broadcast_to(initial_cov, (N, 6, 6)).copy()
        delta = np.zeros((N, 4))
        for cell in regions:
            inside = np.asarray([key == cell for key in keys])
            mask = self.pre['mask'].copy()
            mask[inside] = False
            z, A, ld, blocks = whiten(self.t, mask, self.pre['rssi'], mean, basis, self.radio_parameters)
            _, mu, cov, _, _ = mixture_posterior(z, A, self.device_prior, ld)
            selected = inside & (np.arange(N) >= 5)
            means[selected] = mu
            covariances[selected] = cov
            for ap, ix, L in blocks:
                residual = self.pre['rssi'][ix, ap] - mean[ix, ap] - basis[ix, ap] @ mu
                score = cho_solve((L, True), residual, check_finite=False)
                kernel = np.exp(2 * self.radio_parameters[4 + ap]) * np.exp(-abs(self.t[selected, None] - self.t[None, ix]) / np.exp(self.radio_parameters[8]))
                delta[selected, ap] = kernel @ score
        self.calibration_forecasts = dict(device_mean=means, device_covariance=covariances, short_residual=delta, mode=0)
        diagnostic.update(joint_marginal_device_mean=eta.tolist(), joint_marginal_device_covariance=P.tolist(), initial_device_mean=initial_mean.tolist(), device_prior_PRE_index=5, initial_mixture_posterior_component_weights=component_weight.tolist(), all_eight_initial_device_hypotheses_marginalized_for_one_whole_path=True, device_uncertainty_marginalized=True, RSSI_FORECAST_timing='Offline spatial-region-cavity', rolling_spatial_regions=len(regions), initial_five_recognition_retained=True)
        return (predictions, diagnostic)

def score(a):
    if not (a.output / 'prediction_complete.json').exists():
        raise ValueError('Save every prediction before scoring')
    cohorts = json.loads((a.capture / 'segments.json').read_text())
    results = {}
    labels = {}
    prior = json.loads(a.prior.read_text())
    radio_field = field(a.base, prior)
    for cohort in a.cohorts:
        data = a.base / ('runs/ap_movement_v79_291/experiment/preprocessed' if cohort == 'AP_movement' else 'data/processed_rssi_causal_short_v1')
        result = score_cohort(a.base, a.base / PARENT_RELATIVE, cohorts[cohort], a.output / cohort, data)
        error_by_phone = {}
        all_errors = []
        for s in cohorts[cohort]:
            name = s['device']
            if (cohort, name) not in labels:
                labels[cohort, name] = json.loads((data / name / 'evaluation_labels.json').read_text())
            with np.load(a.output / cohort / s['trajectory_id'] / 'PRE_forecasts.npz', allow_pickle=False) as z:
                pre = {k: z[k].copy() for k in z.files}
            ix = [i for i, oid in enumerate(pre['observation_ids']) if i >= 5 and str(oid) in labels[cohort, name]]
            xy = np.asarray([labels[cohort, name][str(pre['observation_ids'][i])]['gt_xy_m'] for i in ix])
            mean, B = radio_field(xy, pre['motion'][ix])
            locked = prior.get('geometry_locked_device_basis', False)
            if prior.get('calibration_family_whole_path_selection', False):
                diagnostic = json.loads((a.output / cohort / s['trajectory_id'] / 'axial_solver.json').read_text())
                locked = bool(diagnostic['RSSI_forecast_geometry_locked'])
            if locked:
                _, B = radio_field(np.broadcast_to(np.asarray(s['initial_position_xy_m']), xy.shape), pre['motion'][ix])
                B[:, :, 1] = mean / 50
            if prior.get('device_spatial_basis_strength', 1.0) < 1:
                _, initial_B = radio_field(np.broadcast_to(np.asarray(s['initial_position_xy_m']), xy.shape), pre['motion'][ix])
                alpha = prior['device_spatial_basis_strength']
                B[:, :, 2:] = initial_B[:, :, 2:] + alpha * (B[:, :, 2:] - initial_B[:, :, 2:])
            mode = pre['weights'][ix].argmax(-1)
            mu = pre['device_mean'][ix, mode]
            delta = pre['short_residual'][ix, mode]
            predicted = mean + np.einsum('nad,nd->na', B, mu) + delta
            if prior.get('receiver_angular_sigma_db', 0.0) > 0:
                with np.load(a.output / cohort / s['trajectory_id'] / 'receiver_angular_forecast.npz', allow_pickle=False) as fc:
                    if not np.array_equal(fc['observation_ids'], pre['observation_ids']):
                        raise ValueError('Angular forecast IDs differ')
                    saved_rows = json.loads((a.output / cohort / s['trajectory_id'] / 'predictions.json').read_text())
                    if fc['times_ns'].tolist() != [row['timestamp_ns'] for row in saved_rows]:
                        raise ValueError('Angular forecast timestamps differ')
                    gamma = fc['mean_db'][ix]
                    u = fc['query_horizontal'][ix]
                    valid = fc['query_valid_weight'][ix]
                    u0 = fc['initial_horizontal'].copy()
                d = radio_field.centres[None] - xy[:, None]
                bearing = d / np.sqrt(np.square(d).sum(-1) + 1.0)[..., None]
                d0 = radio_field.centres - np.asarray(s['initial_position_xy_m'])
                b0 = d0 / np.sqrt(np.square(d0).sum(-1) + 1.0)[:, None]
                phi = np.stack(((u * bearing).sum(-1), u[:, :, 1] * bearing[:, :, 0] - u[:, :, 0] * bearing[:, :, 1]), -1)
                start_phi = np.stack(((u0 * b0).sum(-1), u0[1] * b0[:, 0] - u0[0] * b0[:, 1]), -1)
                phi -= valid[:, :, None] * start_phi[None]
                predicted += np.einsum('nac,nc->na', phi, gamma)
            if prior.get('receiver_curvature_sigma_db', 0.0) > 0:
                with np.load(a.output / cohort / s['trajectory_id'] / 'receiver_curvature_forecast.npz', allow_pickle=False) as fc:
                    if not np.array_equal(fc['observation_ids'], pre['observation_ids']):
                        raise ValueError('Curvature forecast IDs differ')
                    saved_rows = json.loads((a.output / cohort / s['trajectory_id'] / 'predictions.json').read_text())
                    if fc['times_ns'].tolist() != [row['timestamp_ns'] for row in saved_rows]:
                        raise ValueError('Curvature forecast timestamps differ')
                    gamma = fc['mean_db'][ix]
                    start = fc['initial_mean_db'].copy()
                predicted += gamma[:, None] * (((mean + 60) / 20) ** 2 - ((start[None] + 60) / 20) ** 2)
            error = abs(predicted - pre['rssi'][ix])[pre['mask'][ix]].tolist()
            all_errors.extend(error)
            error_by_phone.setdefault(name, []).extend(error)
        result['RSSI_MAE_db'] = {k: float(np.mean(v)) for k, v in error_by_phone.items()}
        result['pooled_RSSI_MAE_db'] = float(np.mean(all_errors))
        source_field = 'OPPO-only physical-mean spatial GP' if prior.get('source_spatial_GP_model') else 'Finite-radius OPPO field'
        if prior.get('source_RP_contrast_model'):
            source_field = 'OPPO-only smooth RP-contrast-restored finite-radius field'
        if prior.get('joint_source_field_refit_model'):
            source_field = 'OPPO point/contrast-anchored shared spatial field refitted on 43 source trajectories'
        if prior.get('source_map_anisotropic_model'):
            source_field = 'OPPO-only per-AP spatial-metric and decay field fitted by omitted 2m map regions'
        if prior.get('calibration_basis_residual_model') or prior.get('calibration_basis_residual_models'):
            source_field += ' with jointly trained bounded device basis and OPPO-anchored mean' if prior.get('joint_mean_basis_training') else ' with source-trained bounded device-basis residual; selected mean map preserved'
        if prior.get('receiver_curvature_sigma_db', 0.0) > 0:
            source_field += ' with source-fitted single receiver-curvature latent and unchanged original six-dimensional initial components'
        if prior.get('receiver_angular_sigma_db', 0.0) > 0:
            source_field += ' with source-fitted isotropic antenna-response pair; arbitrary game yaw, measured-contributor attitude and explicit missing masks; original six-dimensional initial components unchanged'
        if prior.get('RSSI_history_noise_source_selected'):
            source_field += ' with source-fitted white variance conditioned on strictly past RSSI; current RSSI excluded from its own variance features'
        if prior.get('RSSI_raw_history_noise_source_selected'):
            source_field += ' with source-fitted raw-sample variance conditioned on strictly past measurements before applying the fixed median contributor operator'
        if prior.get('M3_conditional_grid_decoder'):
            source_field += '; M1/M2 forecasts frozen at the captured parent sensor path; conditional-grid M3 does not recalibrate them at its decoded positions'
        result['RSSI_scope'] = source_field + ' at current reference timestamp; offline estimated-region observation-factor omission, conditioned on immutable initial-five support, the complete sensor-predicted path and frozen causal contexts; evaluator GT XY only for scoring'
        write(a.output / cohort / 'complete.json', result)
        results[cohort] = dict(pooled_metrics=result['pooled_metrics'], pooled_RSSI_MAE_db=result['pooled_RSSI_MAE_db'], RSSI_scope=result['RSSI_scope'], shape=shape(a.output / cohort, cohorts[cohort]))
        print(json.dumps(dict(cohort=cohort, pooled_metrics=result['pooled_metrics'], pooled_RSSI_MAE_db=result['pooled_RSSI_MAE_db'])), flush=True)
    write(a.output / 'comparison.json', dict(status='COMPLETE_JOINT_MARGINAL_LOCATION_AND_CAVITY_RSSI_SCORE', results=dict(joint_sensor_path=results), offline=True, later_GT_used_for_prediction=False))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['fit', 'score'], required=True)
    for name in ['base', 'capture', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--cohorts', nargs='+', default=['original_fixed20', 'AP_movement'])
    a = p.parse_args()
    fit(a) if a.stage == 'fit' else score(a)
if __name__ == '__main__':
    main()
