import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular, cho_solve
from scipy.optimize import minimize
from scipy.special import expit
import torch
from base_evaluate import AVAILABLE_CPUS
from temporal_radio import ORIGINAL_LOAD_ACTOR, PARENT_RELATIVE
from engine_utils import write, score_cohort
from trajectory_correlated_cs import CorrelatedPath
from trajectory_score import shape
_FIELD = None
_SPATIAL_FIELDS = {}
_DECODER_FIELDS = {}
_CONTRAST_FIELDS = {}
_SOURCE_REFIT_FIELDS = {}
_ANISOTROPIC_FIELDS = {}
_RESIDUAL_BASIS_FIELDS = {}

class FiniteRadioField:

    def __init__(self, net):
        f = net.field
        if f.orientation_basis or f.source_model != 'idw':
            raise ValueError('This experiment uses the selected non-harmonic IDW field')

        def array(t):
            return t.detach().cpu().numpy().astype(float)
        self.xy = array(f.xy)
        self.mask = array(f.mask)
        self.values = np.stack([array(v) for v in [f.trend, f.residual, f.var, f.visibility]], -1)
        self.centres = array(f.centres_m)
        self.ap = array(f.ap_embedding.weight)
        self.band = array(f.band_embedding(f.band))
        self.layers = [(array(layer.weight), array(layer.bias)) for layer in f.basis_net if isinstance(layer, torch.nn.Linear)]
        self.strength = array(f.AP_strength()) if f.ap_basis_prior else None
        self.radius2 = 0.15 ** 2

    def __call__(self, points, context, derivative=False):
        offset = points[:, None, :] - self.xy[None]
        d2 = np.square(offset).sum(-1) + self.radius2
        w = self.mask[None] / d2[..., None]
        denominator = w.sum(1)
        component = np.einsum('nga,gac->nac', w, self.values) / denominator[..., None]
        mean = component[..., 0] + component[..., 1]
        radius2 = np.square(points[:, None] - self.centres[None]).sum(-1) + 1.0
        logdist = 0.5 * np.log10(radius2)
        visibility = component[..., 3].clip(0.01, 0.99)
        scalar = np.stack((logdist, component[..., 0] / 50, component[..., 1] / 10, np.log1p(component[..., 2]) / 5, visibility), -1)
        n, a = mean.shape
        inputs = np.concatenate((np.broadcast_to(points[:, None] / 10, (n, a, 2)), scalar, np.broadcast_to(self.ap, (n, a, 8)), np.broadcast_to(self.band, (n, a, 2)), np.broadcast_to(context[:, None], (n, a, 32))), -1)
        if derivative:
            dw = -2 * offset[:, :, None, :] / d2[:, :, None, None] * w[..., None]
            dc = np.einsum('ngaz,ngac->nacz', dw, self.values[None] - component[:, None]) / denominator[..., None, None]
            dm = dc[..., 0, :] + dc[..., 1, :]
            loggradient = (points[:, None] - self.centres[None]) / (math.log(10) * radius2[..., None])
            ds = np.stack((loggradient, dc[..., 0, :] / 50, dc[..., 1, :] / 10, dc[..., 2, :] / (1 + component[..., 2, None]) / 5, dc[..., 3, :] * ((component[..., 3] > 0.01) & (component[..., 3] < 0.99))[..., None]), -2)
            directions = np.concatenate((np.broadcast_to(np.eye(2) / 10, (n, a, 2, 2)), ds, np.zeros((n, a, 42, 2))), -2)
        h = inputs
        for j, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if derivative:
                directions = (directions.swapaxes(-1, -2) @ W.T).swapaxes(-1, -2)
            if j < len(self.layers) - 1:
                p = expit(h)
                if derivative:
                    directions *= (p * (1 + h * (1 - p)))[..., None]
                h = h * p
        learned = np.tanh(h)
        if derivative:
            directions *= (1 - learned ** 2)[..., None]
        if self.strength is not None:
            scale = 1 - 0.85 * abs(self.strength)
            learned = scale * learned + self.strength * np.eye(4)
            if derivative:
                directions *= np.asarray(scale)[..., None]
        basis = np.concatenate((np.ones_like(mean)[..., None], mean[..., None] / 50, learned), -1)
        if not derivative:
            return (mean, basis)
        db = np.concatenate((np.zeros_like(dm)[..., None, :], dm[..., None, :] / 50, directions), -2)
        return (mean, basis, dm, db)

    def source_ap(self, points, ap_index, derivative=False):
        ap_index = np.asarray(ap_index, int)
        offset = points[:, None, :] - self.xy[None]
        d2 = np.square(offset).sum(-1) + self.radius2
        mask = self.mask[:, ap_index].T
        values = (self.values[:, :, 0] + self.values[:, :, 1])[:, ap_index].T
        weight = mask / d2
        denominator = weight.sum(1)
        mean = (weight * values).sum(1) / denominator
        if not derivative:
            return mean
        dw = -2 * offset / d2[..., None] * weight[..., None]
        gradient = (dw * (values - mean[:, None])[..., None]).sum(1) / denominator[:, None]
        return (mean, gradient)

    def variance_ap(self, points, ap_index, derivative=False):
        ap_index = np.asarray(ap_index, int)
        offset = points[:, None, :] - self.xy[None]
        d2 = np.square(offset).sum(-1) + self.radius2
        mask = self.mask[:, ap_index].T
        values = self.values[:, ap_index, 2].T
        weight = mask / d2
        denominator = weight.sum(1)
        variance = (weight * values).sum(1) / denominator
        if not derivative:
            return variance
        dw = -2 * offset / d2[..., None] * weight[..., None]
        gradient = (dw * (values - variance[:, None])[..., None]).sum(1) / denominator[:, None]
        return (variance, gradient)

    def basis_inputs_ap(self, points, context, ap_index):
        ap_index = np.asarray(ap_index, int)
        offset = points[:, None, :] - self.xy[None]
        d2 = np.square(offset).sum(-1) + self.radius2
        mask = self.mask[:, ap_index].T
        values = self.values[:, ap_index, :].transpose(1, 0, 2)
        weight = mask / d2
        denominator = weight.sum(1)
        component = np.einsum('mg,mgc->mc', weight, values) / denominator[:, None]
        mean = component[:, 0] + component[:, 1]
        distance2 = np.square(points - self.centres[ap_index]).sum(-1) + 1.0
        scalar = np.stack((0.5 * np.log10(distance2), component[:, 0] / 50, component[:, 1] / 10, np.log1p(component[:, 2]) / 5, component[:, 3].clip(0.01, 0.99)), -1)
        inputs = np.concatenate((points / 10, scalar, self.ap[ap_index], self.band[ap_index], context), -1)
        return (mean, inputs)

    def ap_query(self, points, context, ap_index, derivative=False, return_descriptor=False):
        ap_index = np.asarray(ap_index, int)
        M = len(points)
        offset = points[:, None, :] - self.xy[None]
        d2 = np.square(offset).sum(-1) + self.radius2
        mask = self.mask[:, ap_index].T
        values = self.values[:, ap_index, :].transpose(1, 0, 2)
        w = mask / d2
        denominator = w.sum(1)
        component = np.einsum('mg,mgc->mc', w, values) / denominator[:, None]
        mean = component[:, 0] + component[:, 1]
        centre = self.centres[ap_index]
        radius2 = np.square(points - centre).sum(-1) + 1.0
        visibility = component[:, 3].clip(0.01, 0.99)
        scalar = np.stack((0.5 * np.log10(radius2), component[:, 0] / 50, component[:, 1] / 10, np.log1p(component[:, 2]) / 5, visibility), -1)
        inputs = np.concatenate((points / 10, scalar, self.ap[ap_index], self.band[ap_index], context), -1)
        if derivative:
            dw = -2 * offset / d2[..., None] * w[..., None]
            dc = np.einsum('mgz,mgc->mcz', dw, values - component[:, None]) / denominator[:, None, None]
            dm = dc[:, 0] + dc[:, 1]
            log_gradient = (points - centre) / (math.log(10) * radius2[:, None])
            ds = np.stack((log_gradient, dc[:, 0] / 50, dc[:, 1] / 10, dc[:, 2] / (1 + component[:, 2, None]) / 5, dc[:, 3] * ((component[:, 3] > 0.01) & (component[:, 3] < 0.99))[:, None]), -2)
            directions = np.concatenate((np.broadcast_to(np.eye(2) / 10, (M, 2, 2)), ds, np.zeros((M, 42, 2))), -2)
        input_directions = directions if derivative and return_descriptor else None
        h = inputs
        for j, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if derivative:
                directions = (directions.swapaxes(-1, -2) @ W.T).swapaxes(-1, -2)
            if j < len(self.layers) - 1:
                probability = expit(h)
                if derivative:
                    directions *= (probability * (1 + h * (1 - probability)))[..., None]
                h = h * probability
        learned = np.tanh(h)
        if derivative:
            directions *= (1 - learned ** 2)[..., None]
        if self.strength is not None:
            scale = 1 - 0.85 * abs(self.strength)
            learned = scale * learned + self.strength * np.eye(4)[ap_index]
            if derivative:
                directions *= np.asarray(scale)[..., None]
        basis = np.concatenate((np.ones((M, 1)), mean[:, None] / 50, learned), -1)
        if not derivative:
            return (mean, basis, inputs, None) if return_descriptor else (mean, basis)
        db = np.concatenate((np.zeros((M, 1, 2)), dm[:, None] / 50, directions), -2)
        result = (mean, basis, dm, db)
        return result + (inputs, input_directions) if return_descriptor else result

def _mean_field(base, prior=None):
    global _FIELD
    if _FIELD is None:
        net, _, _, _, _ = ORIGINAL_LOAD_ACTOR(base, base / PARENT_RELATIVE)
        _FIELD = FiniteRadioField(net)
        if AVAILABLE_CPUS:
            os.sched_setaffinity(0, AVAILABLE_CPUS)
    if prior is not None and prior.get('source_map_anisotropic_model'):
        if any((prior.get(k) for k in ['source_RP_contrast_model', 'source_spatial_GP_model', 'field_decoder_override', 'joint_source_field_refit_model'])):
            raise ValueError('Evaluate the OPPO spatial metric separately from other field replacements')
        from trajectory_anisotropic_field import AnisotropicSourceField
        path = str(Path(prior['source_map_anisotropic_model']).resolve())
        if path not in _ANISOTROPIC_FIELDS:
            _ANISOTROPIC_FIELDS[path] = AnisotropicSourceField(_FIELD, path)
        return _ANISOTROPIC_FIELDS[path]
    if prior is not None and prior.get('joint_source_field_refit_model'):
        if any((prior.get(k) for k in ['source_RP_contrast_model', 'source_spatial_GP_model', 'field_decoder_override'])):
            raise ValueError('Evaluate shared source field refitting separately from other field replacements')
        from trajectory_RP_contrast import RPContrastField
        key = (prior['joint_source_field_refit_model'], float(prior.get('joint_source_field_refit_strength', 1.0)))
        if key not in _SOURCE_REFIT_FIELDS:
            _SOURCE_REFIT_FIELDS[key] = RPContrastField(_FIELD, *key)
        return _SOURCE_REFIT_FIELDS[key]
    if prior is not None and prior.get('source_RP_contrast_model'):
        if prior.get('source_spatial_GP_model') or prior.get('field_decoder_override'):
            raise ValueError('Evaluate the source RP contrast mechanism without changing the frozen nonlinear device descriptor')
        from trajectory_RP_contrast import RPContrastField
        key = (prior['source_RP_contrast_model'], float(prior.get('source_RP_contrast_strength', 1.0)))
        if key not in _CONTRAST_FIELDS:
            _CONTRAST_FIELDS[key] = RPContrastField(_FIELD, *key)
        return _CONTRAST_FIELDS[key]
    if prior is not None and prior.get('field_decoder_override'):
        if prior.get('source_spatial_GP_model'):
            raise ValueError('Decoder training uses the preserved IDW source field')
        path = str(Path(prior['field_decoder_override']).resolve())
        if path not in _DECODER_FIELDS:
            adapted = copy.copy(_FIELD)
            layers = []
            with np.load(path, allow_pickle=False) as z:
                for index, (oldW, oldb) in enumerate(_FIELD.layers):
                    W = z['weight_' + str(index)].astype(float)
                    b = z['bias_' + str(index)].astype(float)
                    if W.shape != oldW.shape or b.shape != oldb.shape:
                        raise ValueError('Trained decoder tensor contract changed')
                    layers.append((W, b))
            adapted.layers = layers
            _DECODER_FIELDS[path] = adapted
        return _DECODER_FIELDS[path]
    if prior is not None and prior.get('source_spatial_GP_model'):
        model = prior['source_spatial_GP_model']
        if model not in _SPATIAL_FIELDS:
            from trajectory_radio_gp import SpatialRadioField
            _SPATIAL_FIELDS[model] = SpatialRadioField(_FIELD, model)
        return _SPATIAL_FIELDS[model]
    return _FIELD

def field(base, prior=None):
    selected = _mean_field(base, prior)
    paths = [] if prior is None else list(prior.get('calibration_basis_residual_models', []))
    if prior is not None and prior.get('calibration_basis_residual_model'):
        paths.append(prior['calibration_basis_residual_model'])
    if paths:
        if prior.get('field_decoder_override'):
            raise ValueError('The bounded residual preserves the frozen original basis network')
        from trajectory_calibration_residual import ResidualBasisField
        resolved = [str(Path(path).resolve()) for path in paths]
        if len(set(resolved)) != len(resolved):
            raise ValueError('Repeated calibration residual would be applied twice')
        for path in resolved:
            key = (id(selected), path)
            if key not in _RESIDUAL_BASIS_FIELDS:
                _RESIDUAL_BASIS_FIELDS[key] = ResidualBasisField(selected, path)
            selected = _RESIDUAL_BASIS_FIELDS[key]
    return selected

def read_prior(pre):
    weight = pre['weights'][0].astype(float)
    weight /= weight.sum()
    mu = pre['device_mean'][0].astype(float)
    m = (mu * weight[:, None]).sum(0)
    d = mu - m
    P = ((pre['device_covariance'][0] + d[:, :, None] * d[:, None, :]) * weight[:, None, None]).sum(0)
    P = 0.5 * (P + P.T) + np.eye(6) * 0.0001
    return (m, P)

def radio_blocks(t, mask, parameters):
    blocks = []
    for ap in range(4):
        ix = np.where(mask[:, ap] & (np.arange(len(t)) >= 5))[0]
        if not len(ix):
            continue
        dt = abs(t[ix, None] - t[None, ix])
        white = np.exp(parameters[ap])
        colored = np.exp(parameters[4 + ap])
        tau = np.exp(parameters[8])
        K = colored ** 2 * np.exp(-dt / tau) + np.eye(len(ix)) * (white ** 2 + 0.1 ** 2)
        L = cholesky(K, lower=True, check_finite=False)
        blocks.append((ap, ix, L))
    return blocks

def source_rows(a):
    source = json.loads((a.capture / 'segments.json').read_text())['source_train']
    labels = {}
    rows = []
    radio_field = field(a.base)
    for ordinal, s in enumerate(source):
        name = s['device']
        if name not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target in RSSI likelihood fit')
        if name not in labels:
            labels[name] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / name / 'evaluation_labels.json').read_text())
        folder = a.capture / 'source_train' / s['trajectory_id']
        with np.load(folder / 'PRE_forecasts.npz', allow_pickle=False) as z:
            pre = {k: z[k].copy() for k in z.files}
        predictions = json.loads((folder / 'predictions.json').read_text())
        xy = np.asarray([labels[name][r['observation_id']]['gt_xy_m'] if r['observation_id'] in labels[name] else s['initial_position_xy_m'] for r in predictions])
        mask = pre['mask'].copy()
        valid = np.asarray([r['observation_id'] in labels[name] for r in predictions])
        mask[~valid] = False
        t = np.asarray([(r['timestamp_ns'] - s['start_ns']) / 1000000000.0 for r in predictions])
        keep = np.zeros(len(t), bool)
        previous = -1e+30
        for j, tt in enumerate(t):
            if j >= 5 and tt - previous >= 0.4:
                keep[j] = True
                previous = tt
        mask[~keep] = False
        mean, basis = radio_field(xy, pre['motion'])
        mu, P = read_prior(pre)
        U = cholesky(P, lower=True)
        delta = (pre['short_residual'] * pre['weights'][..., None]).sum(1)
        rows.append(dict(t=t, mask=mask, rssi=pre['rssi'].astype(float), mean=mean, basis=basis, prior_mean=mu, prior_cholesky=U, delta=delta, heldout=ordinal % 5 == 0))
    return rows

def fit_policy(task):
    rows, use_delta = task
    if AVAILABLE_CPUS:
        os.sched_setaffinity(0, AVAILABLE_CPUS)

    def objective(parameters, heldout):
        total = 0.0
        count = 0
        for r in rows:
            if r['heldout'] != heldout:
                continue
            blocks = radio_blocks(r['t'], r['mask'], parameters)
            zs = []
            As = []
            logdet = 0.0
            observed = r['rssi'] - r['mean'] - np.einsum('nad,d->na', r['basis'], r['prior_mean'])
            if use_delta:
                observed = observed - r['delta']
            for ap, ix, L in blocks:
                zs.append(solve_triangular(L, observed[ix, ap], lower=True, check_finite=False))
                As.append(solve_triangular(L, r['basis'][ix, ap] @ r['prior_cholesky'], lower=True, check_finite=False))
                logdet += 2 * np.log(np.diag(L)).sum()
            z = np.concatenate(zs)
            A = np.concatenate(As)
            precision = np.eye(6) + A.T @ A
            chol = cholesky(precision, lower=True, check_finite=False)
            rhs = A.T @ z
            nll = 0.5 * (z @ z - rhs @ cho_solve((chol, True), rhs, check_finite=False) + logdet + 2 * np.log(np.diag(chol)).sum() + len(z) * math.log(2 * math.pi))
            total += nll
            count += len(z)
        return float(total / count)
    initial = np.log([1.5] * 4 + [2.0] * 4 + [2.0])
    bounds = [(math.log(0.2), math.log(12.0))] * 4 + [(math.log(0.02), math.log(15.0))] * 4 + [(math.log(0.1), math.log(30.0))]
    began = time.monotonic()
    result = minimize(objective, initial, args=(False,), bounds=bounds, method='L-BFGS-B', options=dict(maxiter=100, ftol=2e-07))
    answer = dict(use_legacy_short_residual=use_delta, parameters=result.x.tolist(), train_NLL=float(result.fun), heldout_NLL=objective(result.x, True), converged=bool(result.success), termination=str(result.message), seconds=time.monotonic() - began)
    print(json.dumps(answer), flush=True)
    return answer

def fit(a):
    rows = source_rows(a)
    with ProcessPoolExecutor(max_workers=2) as pool:
        fits = list(pool.map(fit_policy, [(rows, False), (rows, True)]))
    best = min(fits, key=lambda r: r['heldout_NLL'])
    prior = json.loads(a.prior.read_text())
    prior.update(joint_RSSI_parameters=best['parameters'], joint_RSSI_use_legacy_short_residual=best['use_legacy_short_residual'], source_RSSI_fits=fits, source_RSSI_noise_selected_by='11 source trajectory head-selection split; no target labels', source_map_IDW_radius_m=0.15, device_prior_from='First frozen PRE initial posterior only', initial_five_RSSI_not_recounted=True, target_GT_in_joint_training_or_selection=False)
    write(a.output, prior)

class JointPath(CorrelatedPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wifi_weight[:] = 0.0
        with np.load(args[0] / 'PRE_forecasts.npz', allow_pickle=False) as z:
            self.pre = {k: z[k].copy() for k in z.files}
        self.radio_field = field(Path(self.prior['joint_runtime_base']))
        self.prior_mu, self.prior_P = read_prior(self.pre)
        self.prior_precision = np.linalg.inv(self.prior_P)
        self.prior_vector = self.prior_precision @ self.prior_mu
        self.delta = (self.pre['short_residual'] * self.pre['weights'][..., None]).sum(1) if self.prior['joint_RSSI_use_legacy_short_residual'] else np.zeros_like(self.pre['rssi'])
        self.radio_blocks = radio_blocks(self.t, self.pre['mask'], np.asarray(self.prior['joint_RSSI_parameters']))

    def radio(self, q, return_posterior=False):
        xy = self.B @ q
        mean, basis, dm, db = self.radio_field(xy, self.pre['motion'], True)
        zs = []
        As = []
        for ap, ix, L in self.radio_blocks:
            zs.append(solve_triangular(L, self.pre['rssi'][ix, ap] - self.delta[ix, ap] - mean[ix, ap], lower=True, check_finite=False))
            As.append(solve_triangular(L, basis[ix, ap], lower=True, check_finite=False))
        z = np.concatenate(zs)
        A = np.concatenate(As)
        precision = self.prior_precision + A.T @ A
        posterior = np.linalg.solve(precision, self.prior_vector + A.T @ z)
        residual = z - A @ posterior
        difference = posterior - self.prior_mu
        loss = 0.5 * float(residual @ residual + difference @ self.prior_precision @ difference)
        if return_posterior:
            return (posterior, np.linalg.inv(precision), loss)
        gradient = np.zeros_like(xy)
        offset = 0
        J = dm + np.einsum('nadz,d->naz', db, posterior)
        for ap, ix, L in self.radio_blocks:
            e = residual[offset:offset + len(ix)]
            offset += len(ix)
            score = solve_triangular(L.T, e, lower=False, check_finite=False)
            gradient[ix] -= score[:, None] * J[ix, ap]
        return (loss, self.B.T @ gradient)

    def objective(self, vector, mode):
        value, gradient = super().objective(vector, mode)
        q, _ = self.unpack(vector)
        loss, grad = self.radio(q)
        value += loss
        gradient[:2 * (self.n - 1)] += grad[1:].ravel()
        return (value, gradient)

    def solve(self):
        seeds = []
        for mode in range(8):
            seeds.append(np.asarray([r['mode_xy_m'][mode] for r in self.predictions]))
        seeds.append(self.learned_path)
        outcomes = []
        for xy in seeds:
            q = np.stack([np.interp(self.knots, self.t, xy[:, j]) for j in range(2)], -1)
            q += self.initial - q[0]
            x = np.r_[q[1:].ravel(), np.zeros(4)]
            result = minimize(self.objective, x, args=(0,), jac=True, method='L-BFGS-B', options=dict(maxiter=250, ftol=1e-08, gtol=0.0002, maxls=30))
            outcomes.append(result)
        result = min(outcomes, key=lambda r: r.fun)
        q, _ = self.unpack(result.x)
        xy = self.B @ q
        mu, P, loss = self.radio(q, True)
        predictions = [dict(p, xy_m=x.tolist(), inference_timing='OFFLINE_JOINT_CALIBRATION_AND_PATH', later_position_GT_used=False) for p, x in zip(self.predictions, xy)]
        diagnostic = dict(iterations=int(result.nit), converged=bool(result.success), termination=str(result.message), objective=float(result.fun), RSSI_profile_objective=loss, device_mean=mu.tolist(), device_covariance=P.tolist(), initial_device_mean=self.prior_mu.tolist(), branch_objectives=[float(r.fun) for r in outcomes])
        return (predictions, diagnostic)

def job(task):
    base, capture, output, learned, cohort, s, prior = task
    base, capture, output, learned = (Path(base), Path(capture), Path(output), Path(learned))
    torch.set_num_threads(1)
    if AVAILABLE_CPUS:
        os.sched_setaffinity(0, AVAILABLE_CPUS)
    source = capture / cohort / s['trajectory_id']
    target = output / cohort / s['trajectory_id']
    from support_label_barrier import deny_label_reads
    with deny_label_reads() as audit:
        rows = json.loads((learned / cohort / s['trajectory_id'] / 'predictions.json').read_text())
        path = np.asarray([r['xy_m'] for r in rows])
        n = len(path)
        solver = JointPath(source, s, prior, (np.zeros(n), np.ones(n), np.zeros(n, bool)), False, path)
        if [r['timestamp_ns'] for r in rows] != solver.times_ns.tolist():
            raise ValueError('Proposal timestamps differ')
        predictions, diagnostic = solver.solve()
        target.mkdir(parents=True, exist_ok=True)
        write(target / 'predictions.json', predictions)
        write(target / 'joint_solver.json', diagnostic)
        pre = solver.pre
        pre['device_mean'][:] = np.asarray(diagnostic['device_mean'])
        pre['device_covariance'][:] = np.asarray(diagnostic['device_covariance'])
        pre['short_residual'][:] = solver.delta[:, None]
        np.savez_compressed(target / 'PRE_forecasts.npz', **pre)
        for name in ['CS_diagnostics.json', 'WiFi_quality_diagnostics.json']:
            shutil.copyfile(str(source / name), str(target / name))
        assert not audit['blocked_label_attempts']
    return dict(trajectory_id=s['trajectory_id'], converged=diagnostic['converged'], labels_opened=False)

def infer(a):
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    prior['joint_runtime_base'] = str(a.base.resolve())
    segments = json.loads((a.capture / 'segments.json').read_text())
    jobs = [(str(a.base), str(a.capture), str(a.output), str(a.learned_path), c, s, prior) for c in a.cohorts for s in segments[c]]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        reports = []
        for r in pool.map(job, jobs):
            reports.append(r)
            if len(reports) % 12 == 0:
                print(json.dumps(dict(completed=len(reports), total=len(jobs), seconds=time.monotonic() - began)), flush=True)
    write(a.output / 'prediction_complete.json', dict(status='COMPLETE_JOINT_DEVICE_AND_TRAJECTORY_INFERENCE', later_GT=False, offline_future_sensors=True, CS_ranges_replaced=False, source_prior=prior, reports=reports, independent_initial_device_channel=True, device_parameters_shared_across_time=True, seconds=time.monotonic() - began))

def score(a):
    if not (a.output / 'prediction_complete.json').exists():
        raise ValueError('Complete all predictions before scoring')
    radio_field = field(a.base)
    segments = json.loads((a.capture / 'segments.json').read_text())
    results = {}
    for cohort in a.cohorts:
        data = a.base / ('runs/ap_movement_v79_291/experiment/preprocessed' if cohort == 'AP_movement' else 'data/processed_rssi_causal_short_v1')
        metrics = score_cohort(a.base, a.base / PARENT_RELATIVE, segments[cohort], a.output / cohort, data)
        labels = {}
        error = {}
        for s in segments[cohort]:
            name = s['device']
            if name not in labels:
                labels[name] = json.loads((data / name / 'evaluation_labels.json').read_text())
            with np.load(a.output / cohort / s['trajectory_id'] / 'PRE_forecasts.npz', allow_pickle=False) as z:
                pre = {k: z[k].copy() for k in z.files}
            ix = [i for i, oid in enumerate(pre['observation_ids']) if i >= 5 and str(oid) in labels[name]]
            xy = np.asarray([labels[name][str(pre['observation_ids'][i])]['gt_xy_m'] for i in ix])
            mean, basis = radio_field(xy, pre['motion'][ix])
            mu = (pre['device_mean'][ix] * pre['weights'][ix, :, None]).sum(1)
            delta = (pre['short_residual'][ix] * pre['weights'][ix, :, None]).sum(1)
            prediction = mean + np.einsum('nad,nd->na', basis, mu) + delta
            error.setdefault(name, []).extend(abs(prediction - pre['rssi'][ix])[pre['mask'][ix]].tolist())
        metrics['RSSI_MAE_db'] = {name: float(np.mean(e)) for name, e in error.items()}
        metrics['pooled_RSSI_MAE_db'] = float(np.mean([e for values in error.values() for e in values]))
        metrics['RSSI_scope'] = 'Offline joint device posterior and 15cm finite-resolution OPPO field; evaluator GT XY only, exact inference field'
        metrics['initial_position_known_once'] = True
        write(a.output / cohort / 'complete.json', metrics)
        results[cohort] = dict(pooled_metrics=metrics['pooled_metrics'], pooled_RSSI_MAE_db=metrics['pooled_RSSI_MAE_db'], shape=shape(a.output / cohort, segments[cohort]))
        print(json.dumps(dict(cohort=cohort, **{k: v for k, v in results[cohort].items() if k != 'shape'}, shape={k: v for k, v in results[cohort]['shape'].items() if k != 'per_trajectory'})), flush=True)
    write(a.output / 'comparison.json', dict(status='COMPLETE_OFFLINE_JOINT_CALIBRATION_COMPARISON', results=results, source_fit_only=True, later_position_GT_in_solver=False, scoring_exclusions_changed=False))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['fit', 'infer', 'score'], required=True)
    for name in ['base', 'capture', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--learned-path', type=Path)
    p.add_argument('--cohorts', nargs='+', default=['source_train'])
    p.add_argument('--workers', type=int, default=24)
    a = p.parse_args()
    torch.set_num_threads(1)
    if AVAILABLE_CPUS:
        os.sched_setaffinity(0, AVAILABLE_CPUS)
    if a.stage == 'fit':
        fit(a)
    elif a.stage == 'infer':
        infer(a)
    else:
        score(a)
if __name__ == '__main__':
    main()
