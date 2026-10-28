import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from engine_utils import write
from trajectory_joint_marginal import mixture_posterior
from trajectory_spatial_residual import source_rows, precision_score
from trajectory_temporal_radio import TemporalJointPath

def noise(model, variance, p):
    cov, shared, derivatives = model.noise(p[:13])
    extra = []
    for ap in range(4):
        S = model.S[ap]
        sl = slice(model.bounds[ap], model.bounds[ap + 1])
        value = math.exp(p[13 + ap]) * (S * variance[sl][None]) @ S.T
        cov[ap] += value
        shared[ap] += value
        extra.append(value)
    return (cov, shared, derivatives, extra)

def radio_energy(model, points, radio_field, device_prior, p, derivative=True):
    geometry = model.geometry(points, radio_field, derivative)
    mean, basis = geometry[:2]
    output = radio_field.variance_ap(points, model.ap, derivative)
    variance, dv = output if derivative else (output, None)
    cov, _, _, _ = noise(model, variance, p)
    blocks = model.blocks(cov)
    z, A, ld = model.whiten(mean, basis, blocks)
    loss, eta, P, _, weight = mixture_posterior(z, A, device_prior, ld)
    if not derivative:
        return float(loss)
    dm, db = geometry[2:]
    gm = np.zeros_like(mean)
    gb = np.zeros_like(basis)
    raw_g = np.zeros_like(points)
    for ap, ix, L, score, W, G in precision_score(model, mean, basis, blocks, eta, P):
        gm[ix, ap] = -score
        gb[ix, ap] = W @ P - score[:, None] * eta[None]
        S = model.S[ap][ix]
        diagonal = np.einsum('ij,ij->j', S, G @ S)
        sl = slice(model.bounds[ap], model.bounds[ap + 1])
        raw_g[sl] = math.exp(p[13 + ap]) * diagonal[:, None] * dv[sl]
    raw_gm = np.concatenate([model.S[ap].T @ gm[:, ap] for ap in range(4)])
    raw_gb = np.concatenate([model.S[ap].T @ gb[:, ap] for ap in range(4)])
    raw_g += raw_gm[:, None] * dm + np.einsum('nd,ndz->nz', raw_gb, db)
    return (float(loss), raw_g, (eta, P, weight))

def fit(a):
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    rows, radio_field = source_rows(a, prior)
    for r in rows:
        r['variance'] = radio_field.variance_ap(r['points'], r['model'].ap)

    def objective(p, heldout, derivative=True):
        total = 0.0
        count = 0
        gradient = np.zeros(17)
        for r in rows:
            if r['heldout'] != heldout:
                continue
            model = r['model']
            cov, _, dC, extra = noise(model, r['variance'], p)
            blocks = model.blocks(cov)
            z, A, ld = model.whiten(r['mean'], r['basis'], blocks)
            loss, eta, P, _, _ = mixture_posterior(z, A, r['prior'], ld)
            total += loss
            count += len(z)
            if derivative:
                for ap, ix, L, score, W, G in precision_score(model, r['mean'], r['basis'], blocks, eta, P):
                    for index, dc in zip([ap, 4 + ap, 8, 9 + ap, 13 + ap], list(dC[ap]) + [extra[ap]]):
                        gradient[index] += float((G * dc[np.ix_(ix, ix)]).sum())
        return (total / count, gradient / count) if derivative else float(total / count)
    baseline = np.r_[prior['temporal_RSSI_parameters'], np.full(4, -30.0)]
    initial = np.r_[prior['temporal_RSSI_parameters'], np.full(4, math.log(0.1))]
    d = np.random.RandomState(82602).normal(size=17)
    d /= np.linalg.norm(d)
    h = 0.0001
    _, g = objective(initial, False)
    numeric = (objective(initial + h * d, False, False) - objective(initial - h * d, False, False)) / (2 * h)
    parameter_error = float(abs(numeric - g @ d))
    if parameter_error > 3e-05:
        raise ValueError('OPPO variance coefficient derivative mismatch')
    bounds = [(math.log(0.2), math.log(12.0))] * 4 + [(math.log(0.02), math.log(15.0))] * 4 + [(math.log(0.1), math.log(30.0))] + [(math.log(0.02), math.log(12.0))] * 4 + [(math.log(0.0001), math.log(10.0))] * 4
    result = minimize(objective, initial, args=(False, True), jac=True, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=100, ftol=2e-07))
    if not np.isfinite(result.fun):
        raise FloatingPointError('Nonfinite OPPO map-variance fit')
    validation = objective(result.x, True, False)
    old = objective(baseline, True, False)
    selected = validation < old
    r = rows[0]
    d = np.random.RandomState(82603).normal(size=r['points'].shape)
    d /= np.linalg.norm(d)
    h = 1e-05
    _, g, _ = radio_energy(r['model'], r['points'], radio_field, r['prior'], result.x, True)
    numeric = (radio_energy(r['model'], r['points'] + h * d, radio_field, r['prior'], result.x, False) - radio_energy(r['model'], r['points'] - h * d, radio_field, r['prior'], result.x, False)) / (2 * h)
    position_error = float(abs(numeric - (g * d).sum()))
    if position_error > 0.0001:
        raise ValueError('OPPO variance position derivative mismatch')
    prior.update(RSSI_map_variance_covariance=bool(selected), map_variance_RSSI_parameters=result.x.tolist(), map_variance_RSSI_source_heldout_NLL=validation, map_variance_RSSI_baseline_source_heldout_NLL=old, map_variance_RSSI_source_train_NLL=float(result.fun), map_variance_RSSI_parameter_gradient_difference=parameter_error, map_variance_RSSI_position_gradient_difference=position_error, map_variance_RSSI_target_labels_used=False, map_variance_RSSI_fit_seconds=time.monotonic() - began)
    if selected:
        prior['temporal_RSSI_parameters'] = result.x[:13].tolist()
    write(a.output, prior)
    print(json.dumps(dict(selected=selected, source_heldout_NLL=validation, baseline_source_heldout_NLL=old, OPPO_variance_coefficients=np.exp(result.x[13:17]).tolist(), position_gradient_difference=position_error, seconds=time.monotonic() - began)), flush=True)

class MapVariancePath(TemporalJointPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variance_parameters = np.asarray(self.prior['map_variance_RSSI_parameters'])

    def radio(self, q, mode):
        loss, gradient, posterior = radio_energy(self.observations, self.raw_spline @ q, self.radio_field, self.device_prior, self.variance_parameters, True)
        return (loss, self.raw_spline.T @ gradient, posterior)

    def calibrate_solution(self, predictions, diagnostic):
        q, _ = self.unpack(self.last_selected_vector[:-1])
        variance = self.radio_field.variance_ap(self.raw_spline @ q, self.observations.ap)
        self.radio_covariances, self.radio_shared_covariances, _, _ = noise(self.observations, variance, self.variance_parameters)
        self.radio_blocks = self.observations.blocks(self.radio_covariances)
        predictions, diagnostic = super().calibrate_solution(predictions, diagnostic)
        diagnostic.update(RSSI_map_variance_covariance=True, RSSI_covariance_logdet_and_position_derivative_used=True, OPPO_map_variance_coefficients=np.exp(self.variance_parameters[13:17]).tolist())
        return (predictions, diagnostic)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'queries', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    fit(p.parse_args())
if __name__ == '__main__':
    main()
