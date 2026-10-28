import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from engine_utils import write
from trajectory_joint_calibration import field
from trajectory_joint_marginal import initial_prior, mixture_posterior
from trajectory_temporal_radio import ObservationModel, TemporalJointPath

def noise(model, points, parameters):
    cov, shared, base_derivatives = model.noise(parameters[:13])
    space_derivatives = []
    kernels = []
    ell2 = math.exp(2 * parameters[17])
    for ap in range(4):
        sl = slice(model.bounds[ap], model.bounds[ap + 1])
        x = points[sl]
        delta = x[:, None] - x[None]
        d2 = np.square(delta).sum(-1)
        K = np.exp(-0.5 * d2 / ell2)
        variance = math.exp(2 * parameters[13 + ap])
        S = model.S[ap]
        extra = variance * (S @ K @ S.T)
        cov[ap] += extra
        shared[ap] += extra
        space_derivatives.append((2 * extra, variance * (S @ (K * d2 / ell2) @ S.T)))
        kernels.append((K, delta, variance / ell2))
    return (cov, shared, base_derivatives, space_derivatives, kernels)

def precision_score(model, mean, basis, blocks, eta, P):
    for ap, ix, L in blocks:
        residual = model.pre['rssi'][ix, ap] - mean[ix, ap] - basis[ix, ap] @ eta
        inverse = cho_solve((L, True), np.eye(len(ix)), check_finite=False)
        score = inverse @ residual
        W = inverse @ basis[ix, ap]
        G = 0.5 * (inverse - np.outer(score, score) - W @ P @ W.T)
        yield (ap, ix, L, score, W, G)

def radio_energy(model, points, radio_field, device_prior, parameters, derivative=True):
    geometry = model.geometry(points, radio_field, derivative)
    mean, basis = geometry[:2]
    cov, shared, _, _, kernels = noise(model, points, parameters)
    blocks = model.blocks(cov)
    z, A, ld = model.whiten(mean, basis, blocks)
    loss, eta, P, _, weight = mixture_posterior(z, A, device_prior, ld)
    if not derivative:
        return float(loss)
    dm, db = geometry[2:]
    gm = np.zeros_like(mean)
    gb = np.zeros_like(basis)
    raw_covariance_gradient = np.zeros_like(points)
    for ap, ix, L, score, W, G in precision_score(model, mean, basis, blocks, eta, P):
        gm[ix, ap] = -score
        gb[ix, ap] = W @ P - score[:, None] * eta[None]
        S = model.S[ap][ix]
        raw_G = S.T @ G @ S
        K, delta, scale = kernels[ap]
        sl = slice(model.bounds[ap], model.bounds[ap + 1])
        raw_covariance_gradient[sl] = -2 * scale * np.einsum('ij,ijz->iz', raw_G * K, delta)
    raw_gm = np.concatenate([model.S[ap].T @ gm[:, ap] for ap in range(4)])
    raw_gb = np.concatenate([model.S[ap].T @ gb[:, ap] for ap in range(4)])
    gradient = raw_gm[:, None] * dm + np.einsum('nd,ndz->nz', raw_gb, db) + raw_covariance_gradient
    return (float(loss), gradient, (eta, P, weight))

def source_rows(a, prior):
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    labels = {}
    rows = []
    radio_field = field(a.base, prior)
    for ordinal, s in enumerate(segments):
        name = s['device']
        if name not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target label in spatial propagation noise fitting')
        if name not in labels:
            labels[name] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / name / 'evaluation_labels.json').read_text())
        folder = a.capture / 'source_train' / s['trajectory_id']
        with np.load(folder / 'PRE_forecasts.npz', allow_pickle=False) as z:
            pre = {k: z[k].copy() for k in z.files}
        predictions = json.loads((folder / 'predictions.json').read_text())
        selected = [p for p in predictions if p['observation_id'] in labels[name]]
        t = np.r_[0.0, [(p['timestamp_ns'] - s['start_ns']) / 1000000000.0 for p in selected]]
        xy = np.vstack((s['initial_position_xy_m'], [labels[name][p['observation_id']]['gt_xy_m'] for p in selected]))
        model = ObservationModel(a.queries / 'source_train' / s['trajectory_id'] / 'radio_queries.npz', pre)
        model.mask &= np.asarray([str(oid) in labels[name] for oid in pre['observation_ids']])[:, None]
        points = np.stack([np.interp(model.raw_times.clip(0, t[-1]), t, xy[:, j]) for j in range(2)], -1)
        mean, basis = model.geometry(points, radio_field)
        rows.append(dict(model=model, points=points, mean=mean, basis=basis, prior=initial_prior(pre), heldout=ordinal % 5 == 0))
    return (rows, radio_field)

def fit(a):
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    rows, radio_field = source_rows(a, prior)

    def objective(parameters, heldout, derivative=True):
        total = 0.0
        count = 0
        gradient = np.zeros(18)
        for row in rows:
            if row['heldout'] != heldout:
                continue
            model = row['model']
            cov, _, dC, ds, _ = noise(model, row['points'], parameters)
            blocks = model.blocks(cov)
            z, A, ld = model.whiten(row['mean'], row['basis'], blocks)
            loss, eta, P, _, _ = mixture_posterior(z, A, row['prior'], ld)
            total += loss
            count += len(z)
            if derivative:
                for ap, ix, L, score, W, G in precision_score(model, row['mean'], row['basis'], blocks, eta, P):
                    for index, dc in zip([ap, 4 + ap, 8, 9 + ap, 13 + ap, 17], list(dC[ap]) + list(ds[ap])):
                        gradient[index] += float((G * dc[np.ix_(ix, ix)]).sum())
        return (total / count, gradient / count) if derivative else float(total / count)
    baseline = np.r_[prior['temporal_RSSI_parameters'], np.full(4, math.log(1e-08)), 0.0]
    initial = np.r_[prior['temporal_RSSI_parameters'], np.full(4, math.log(1.0)), 0.0]
    direction = np.random.RandomState(82571).normal(size=18)
    direction /= np.linalg.norm(direction)
    h = 0.0001
    _, g = objective(initial, False)
    numeric = (objective(initial + h * direction, False, False) - objective(initial - h * direction, False, False)) / (2 * h)
    parameter_difference = float(abs(numeric - g @ direction))
    if parameter_difference > 3e-05:
        raise ValueError('Spatial residual covariance parameter derivative mismatch')
    bounds = [(math.log(0.2), math.log(12.0))] * 4 + [(math.log(0.02), math.log(15.0))] * 4 + [(math.log(0.1), math.log(30.0))] + [(math.log(0.02), math.log(12.0))] * 8 + [(math.log(0.2), math.log(8.0))]
    result = minimize(objective, initial, args=(False, True), jac=True, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=100, ftol=2e-07))
    if not np.isfinite(result.fun):
        raise FloatingPointError('Nonfinite spatial propagation fit')
    val = objective(result.x, True, False)
    old = objective(baseline, True, False)
    selected = val < old
    r = rows[0]
    d = np.random.RandomState(82572).normal(size=r['points'].shape)
    d /= np.linalg.norm(d)
    h = 1e-05
    _, gradient, _ = radio_energy(r['model'], r['points'], radio_field, r['prior'], result.x, True)
    numeric = (radio_energy(r['model'], r['points'] + h * d, radio_field, r['prior'], result.x, False) - radio_energy(r['model'], r['points'] - h * d, radio_field, r['prior'], result.x, False)) / (2 * h)
    position_difference = float(abs(numeric - (gradient * d).sum()))
    if position_difference > 0.0001:
        raise ValueError('Spatial covariance position derivative mismatch')
    prior.update(RSSI_spatial_residual_covariance=bool(selected), spatial_RSSI_parameters=result.x.tolist(), spatial_RSSI_source_train_NLL=float(result.fun), spatial_RSSI_source_heldout_NLL=val, spatial_RSSI_baseline_source_heldout_NLL=old, spatial_RSSI_parameter_gradient_difference=parameter_difference, spatial_RSSI_position_gradient_difference=position_difference, spatial_RSSI_fit_converged=bool(result.success), spatial_RSSI_fit_termination=str(result.message), spatial_RSSI_target_labels_used=False, spatial_RSSI_complete_covariance_logdet_and_position_derivative=True, spatial_RSSI_fit_seconds=time.monotonic() - began)
    if selected:
        prior['temporal_RSSI_parameters'] = result.x[:13].tolist()
    write(a.output, prior)
    print(json.dumps(dict(selected=selected, source_heldout_NLL=val, baseline_source_heldout_NLL=old, spatial_sigma_db=np.exp(result.x[13:17]).tolist(), spatial_lengthscale_m=math.exp(result.x[17]), temporal_sigma_db=np.exp(result.x[4:8]).tolist(), position_gradient_difference=position_difference, seconds=time.monotonic() - began)), flush=True)

class SpatialResidualPath(TemporalJointPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_parameters = np.asarray(self.prior['spatial_RSSI_parameters'])

    def radio(self, q, mode):
        loss, raw_gradient, posterior = radio_energy(self.observations, self.raw_spline @ q, self.radio_field, self.device_prior, self.spatial_parameters, True)
        return (loss, self.raw_spline.T @ raw_gradient, posterior)

    def calibrate_solution(self, predictions, diagnostic):
        q, _ = self.unpack(self.last_selected_vector[:-1])
        self.radio_covariances, self.radio_shared_covariances, _, _, _ = noise(self.observations, self.raw_spline @ q, self.spatial_parameters)
        self.radio_blocks = self.observations.blocks(self.radio_covariances)
        predictions, diagnostic = super().calibrate_solution(predictions, diagnostic)
        diagnostic.update(RSSI_spatial_residual_covariance=True, RSSI_covariance_logdet_and_position_derivative_used=True, spatial_RSSI_sigma_db=np.exp(self.spatial_parameters[13:17]).tolist(), spatial_RSSI_lengthscale_m=math.exp(self.spatial_parameters[17]))
        return (predictions, diagnostic)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'queries', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    fit(p.parse_args())
if __name__ == '__main__':
    main()
