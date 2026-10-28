import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
import torch
from base_evaluate import AVAILABLE_CPUS
from engine_utils import write
from trajectory_path import SensorPath

def innovation_nll(residual, L, pi, gradient=False):
    z = solve_triangular(L, residual, lower=True, check_finite=False)
    diagonal = np.diag(L)
    normal = -0.5 * (z * z + math.log(2 * math.pi)) + np.log1p(-pi)
    scale = 10.0 / diagonal
    heavy = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi) - np.log(scale) - 2 * np.log1p(z * z / (3 * scale * scale)) + np.log(pi)
    logp = np.logaddexp(normal, heavy)
    loss = float((-logp + np.log(diagonal)).sum())
    if not gradient:
        return loss
    responsibility = np.exp(normal - logp)
    derivative = responsibility * z + (1 - responsibility) * 4 * z / (3 * scale * scale + z * z)
    return (loss, solve_triangular(L.T, derivative, lower=False, check_finite=False))

def covariance(parameters, dt, sigma, past_mass, past_time_moment=None):
    white, colored, constant, tau = np.exp(parameters[:4])
    result = np.diag((white * sigma) ** 2 + 0.05 ** 2)
    result += colored ** 2 * np.exp(-dt / tau) + constant ** 2
    result += 0.555741316 ** 2 * np.outer(past_mass, past_mass)
    if len(parameters) > 8:
        if past_time_moment is None:
            raise ValueError('Missing physical history time uncertainty')
        result += math.exp(2 * parameters[8]) * np.outer(past_time_moment, past_time_moment)
    return result

def fit(a):
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    lags = np.asarray(prior['CS_response_lags_s'])
    weights = np.asarray(prior['CS_response_weights'])
    labels = {}
    blocks = []
    for ordinal, s in enumerate(segments):
        name = s['device']
        if name not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target label in temporal noise fit')
        if name not in labels:
            labels[name] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / name / 'evaluation_labels.json').read_text())
        folder = a.capture / 'source_train' / s['trajectory_id']
        pred = json.loads((folder / 'predictions.json').read_text())
        rows = [r for r in pred if r['observation_id'] in labels[name]]
        t = np.r_[0.0, [(r['timestamp_ns'] - s['start_ns']) / 1000000000.0 for r in rows]]
        xy = np.vstack((s['initial_position_xy_m'], [labels[name][r['observation_id']]['gt_xy_m'] for r in rows]))
        with np.load(folder / 'CS_training_contexts.npz', allow_pickle=False) as z:
            cs = {k: z[k].copy() for k in z.files}
        with np.load(folder / 'trajectory_factors.npz', allow_pickle=False) as z:
            f = {k: z[k].copy() for k in z.files}
        ct = (cs['timestamp_ns'] - s['start_ns']) / 1000000000.0
        query = ct[:, None] - lags[None]
        event_weights = np.broadcast_to(weights, query.shape)
        past_moment = None
        if prior.get('CS_actual_history_query_root'):
            from trajectory_cs_history_response import load_history
            h = load_history(prior, folder, cs['observation_ids'])
            query = h['history_times_s']
            event_weights = h['response_weights']
            past_moment = h['past_time_moment']
        points = np.stack([np.interp(query.clip(0, t[-1]), t, xy[:, j]) for j in range(2)], -1)
        anchors = f['anchors'][cs['anchor_index']]
        expected = (np.linalg.norm(points - anchors[:, None], axis=-1) * event_weights).sum(-1)
        for anchor in range(4):
            ix = np.where((cs['anchor_index'] == anchor) & (ct <= t[-1]))[0]
            if not len(ix):
                continue
            blocks.append(dict(anchor=anchor, heldout=ordinal % 5 == 0, residual=cs['range_m'][ix] - expected[ix], dt=abs(ct[ix, None] - ct[None, ix]), sigma=f['cs_sigma'][ix], pi=np.clip(f['cs_pi'][ix], 1e-05, 1 - 1e-05), past_mass=((query[ix] < 0) * event_weights[ix]).sum(-1), past_time_moment=None if past_moment is None else past_moment[ix]))

    def objective(parameters, heldout):
        total = 0.0
        count = 0
        for b in blocks:
            if b['heldout'] != heldout:
                continue
            K = covariance(parameters, b['dt'], b['sigma'], b['past_mass'], b['past_time_moment'])
            L = cholesky(K, lower=True, check_finite=False)
            total += innovation_nll(b['residual'] - parameters[4 + b['anchor']], L, b['pi'])
            count += len(b['residual'])
        return total / count
    mean = [float(np.median(np.concatenate([b['residual'] for b in blocks if not b['heldout'] and b['anchor'] == j]))) for j in range(4)]
    initial = np.r_[np.log([0.35, 0.7, 0.8, 1.5]), mean]
    bounds = [(math.log(0.02), math.log(4.0)), (math.log(0.005), math.log(6.0)), (math.log(0.01), math.log(6.0)), (math.log(0.05), math.log(30.0))] + [(-5.0, 5.0)] * 4
    if prior.get('CS_actual_history_query_root'):
        initial = np.r_[initial, math.log(0.7)]
        bounds.append((math.log(0.02), math.log(3.0)))
    result = minimize(objective, initial, args=(False,), bounds=bounds, method='L-BFGS-B', options=dict(maxiter=150, ftol=1e-08))
    if not np.isfinite(result.fun):
        raise FloatingPointError('Temporal CS source likelihood did not finish')
    parameters = result.x
    independent = parameters.copy()
    independent[1] = math.log(0.005)

    def reduced(p, heldout):
        value = independent.copy()
        value[selection] = p
        return objective(value, heldout)
    selection = [0, 2, 4, 5, 6, 7] + ([8] if len(parameters) > 8 else [])
    reduced_fit = minimize(reduced, independent[selection], args=(False,), bounds=[bounds[j] for j in selection], method='L-BFGS-B', options=dict(maxiter=120, ftol=1e-08))
    independent[selection] = reduced_fit.x
    prior.update(CS_temporal_innovation_parameters=parameters.tolist(), CS_temporal_covariance_model='white quality variance + OU colored error + constant latent anchor offset', CS_temporal_source_train_NLL=objective(parameters, False), CS_temporal_source_heldout_NLL=objective(parameters, True), CS_independent_source_heldout_NLL=objective(independent, True), CS_temporal_fit_converged=bool(result.success), CS_temporal_fit_message=str(result.message), CS_quality_outlier_probability_interface_unchanged=True, CS_ranges_replaced=False, CS_cached_online_bias_reused=False, CS_inference_GT_used=False, source_only_noise_fit=True, source_noise_fit_seconds=time.monotonic() - began)
    write(a.output, prior)
    print(json.dumps(dict(train_NLL=prior['CS_temporal_source_train_NLL'], heldout_NLL=prior['CS_temporal_source_heldout_NLL'], independent_heldout_NLL=prior['CS_independent_source_heldout_NLL'], white_multiplier=float(np.exp(parameters[0])), colored_sigma_m=float(np.exp(parameters[1])), constant_sigma_m=float(np.exp(parameters[2])), correlation_time_s=float(np.exp(parameters[3])), anchor_mean_m=parameters[4:8].tolist(), past_radial_velocity_sigma_m_s=float(np.exp(parameters[8])) if len(parameters) > 8 else None, seconds=time.monotonic() - began)), flush=True)

class CorrelatedPath(SensorPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.cs_temporal_response:
            raise ValueError('Corrector response is required')
        self.cs_weight[:] = 0.0
        self.parameters = np.asarray(self.prior['CS_temporal_innovation_parameters'])
        self.blocks = []
        with np.load(args[0] / 'trajectory_factors.npz', allow_pickle=False) as z:
            sigma = z['cs_sigma'].copy()
        for anchor in range(4):
            ix = np.where(self.anchor_index == anchor)[0]
            if not len(ix):
                continue
            dt = abs(self.ct[ix, None] - self.ct[None, ix])
            K = covariance(self.parameters, dt, sigma[ix], self.cs_past_mass[ix], None if self.cs_past_time_moment is None else self.cs_past_time_moment[ix])
            self.blocks.append((anchor, ix, cholesky(K, lower=True, check_finite=False)))

    def objective(self, vector, mode):
        value, gradient = super().objective(vector, mode)
        q, _ = self.unpack(vector)
        points = np.einsum('mln,nd->mld', self.C_history, q)
        delta = points - self.anchor[:, None]
        distance = np.sqrt((delta * delta).sum(-1) + 1e-08)
        expected = (distance * self.cs_response_weights).sum(-1)
        residual = self.cs_range - expected - self.parameters[4:][self.anchor_index]
        derivative = np.zeros_like(residual)
        for anchor, ix, L in self.blocks:
            loss, d = innovation_nll(residual[ix], L, self.pi[ix], True)
            value += loss
            derivative[ix] = -d
        grad = np.einsum('mln,mld,ml,m->nd', self.C_history, delta / distance[..., None], self.cs_response_weights, derivative)
        gradient[:2 * (self.n - 1)] += grad[1:].ravel()
        return (value, gradient)

def job(task):
    capture, output, learned_root, cohort, s, prior = task
    capture, output, learned_root = (Path(capture), Path(output), Path(learned_root))
    torch.set_num_threads(1)
    if AVAILABLE_CPUS:
        os.sched_setaffinity(0, AVAILABLE_CPUS)
    source = capture / cohort / s['trajectory_id']
    target = output / cohort / s['trajectory_id']
    from support_label_barrier import deny_label_reads
    with deny_label_reads() as audit:
        learned = json.loads((learned_root / cohort / s['trajectory_id'] / 'predictions.json').read_text())
        path = np.asarray([r['xy_m'] for r in learned])
        n = len(path)
        speed = (np.zeros(n), np.ones(n), np.zeros(n, bool))
        solver = CorrelatedPath(source, s, prior, speed, False, path)
        if [r['timestamp_ns'] for r in learned] != solver.times_ns.tolist():
            raise ValueError('Proposal times differ')
        predictions, diagnostic = solver.solve()
        target.mkdir(parents=True, exist_ok=True)
        write(target / 'predictions.json', predictions)
        for name in ['PRE_forecasts.npz', 'CS_diagnostics.json', 'WiFi_quality_diagnostics.json']:
            shutil.copyfile(str(source / name), str(target / name))
        diagnostic.update(trajectory_id=s['trajectory_id'], labels_opened=False, CS_ranges_replaced=False, known_start_only=True, offline_future_sensors_used=True)
        write(target / 'trajectory_solver.json', diagnostic)
        assert not audit['blocked_label_attempts']
    return diagnostic

def infer(a):
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    if prior['CS_temporal_source_heldout_NLL'] >= prior['CS_independent_source_heldout_NLL']:
        raise ValueError('Temporal CS likelihood did not improve the source heldout likelihood')
    segments = json.loads((a.capture / 'segments.json').read_text())
    jobs = [(str(a.capture), str(a.output), str(a.learned_path), c, s, prior) for c in a.cohorts for s in segments[c]]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        reports = []
        for r in pool.map(job, jobs):
            reports.append(r)
            if len(reports) % 12 == 0:
                print(json.dumps(dict(completed=len(reports), total=len(jobs), seconds=time.monotonic() - began)), flush=True)
    write(a.output / 'prediction_complete.json', dict(status='COMPLETE_CORRELATED_CS_SENSOR_ONLY_PATHS', reports=reports, later_GT=False, offline_future_sensors=True, CS_ranges_replaced=False, source_prior=prior, seconds=time.monotonic() - began))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['fit', 'infer'], required=True)
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
    else:
        infer(a)
if __name__ == '__main__':
    main()
