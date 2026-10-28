import argparse
import concurrent.futures
import fcntl
import functools
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from scipy.linalg import solve_triangular
BASE = Path('.')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from trajectory_cross_region import CrossRegionPath
from trajectory_temporal_radio import TemporalJointPath
from trajectory_joint_marginal import mixture_posterior

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def self_predictive(z, A, prior, ld, gradient=True):
    zz = torch.tensor(z, dtype=torch.float64, requires_grad=gradient)
    aa = torch.tensor(A, dtype=torch.float64, requires_grad=gradient)

    def tensor(key):
        return torch.tensor(prior[key], dtype=torch.float64)
    precision = tensor('precision') + aa.T @ aa
    chol = torch.linalg.cholesky(precision)
    cov = torch.cholesky_inverse(chol)
    rhs = tensor('vector') + aa.T @ zz
    mu = torch.cholesky_solve(rhs[..., None], chol).squeeze(-1)
    error = zz[None] - mu @ aa.T
    delta = mu - tensor('mean')
    component_loss = 0.5 * (error.square().sum(-1) + torch.einsum('ki,kij,kj->k', delta, tensor('precision'), delta) + tensor('logdet') + 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1))
    log_weight = (tensor('log_weight') - component_loss).log_softmax(-1)
    predictive_chol = torch.linalg.cholesky(precision + aa.T @ aa)
    b = error @ aa
    quadratic = error.square().sum(-1) - (b * torch.cholesky_solve(b[..., None], predictive_chol).squeeze(-1)).sum(-1)
    logdet = -2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1) + 2 * predictive_chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    loss = -torch.logsumexp(log_weight - 0.5 * (quadratic + logdet + ld + len(z) * math.log(2 * math.pi)), 0)
    mean = (log_weight.exp()[:, None] * mu).sum(0)
    audit = dict(posterior_information_added_once=True, posterior_mean=mean.detach().numpy(), posterior_precision=precision.detach().numpy())
    if gradient:
        gz, gA = torch.autograd.grad(loss, (zz, aa))
        return (float(loss.detach()), gz.detach().numpy(), gA.detach().numpy(), audit)
    return (float(loss.detach()), audit)

class M2ControlPath(CrossRegionPath):

    def radio(self, q, mode):
        policy = self.prior['ablation_M2_score']
        if policy == 'cross_region':
            return super().radio(q, mode)
        if policy == 'global_marginal':
            return TemporalJointPath.radio(self, q, mode)
        if policy != 'self_predictive':
            raise ValueError('Unknown M2 scoring intervention')
        model = self.observations
        mean, B, dm, db = model.geometry(self.raw_spline @ q, self.radio_field, True)
        z, A, ld = model.whiten(mean, B, self.radio_blocks)
        loss, gz, gA, _ = self_predictive(z, A, self.device_prior, ld)
        gm = np.zeros_like(mean)
        gb = np.zeros_like(B)
        offset = 0
        for ap, ix, L in self.radio_blocks:
            end = offset + len(ix)
            gm[ix, ap] = -solve_triangular(L.T, gz[offset:end], lower=False, check_finite=False)
            gb[ix, ap] = solve_triangular(L.T, gA[offset:end], lower=False, check_finite=False)
            offset = end
        _, eta, P, _, weights = mixture_posterior(z, A, self.device_prior, ld)
        return (loss, model.pullback(gm, gb, dm, db, self.raw_spline), (eta, P, weights))

    def axial_objective(self, *args, **kwargs):
        self._calls = getattr(self, '_calls', 0) + 1
        return super().axial_objective(*args, **kwargs)

    def solve_axis(self, alternative, modes):
        start = time.monotonic()
        before = getattr(self, '_calls', 0)
        pred, diag = super().solve_axis(alternative, modes)
        diag.update(ablation_M2_score=self.prior['ablation_M2_score'], ablation_objective_evaluations=getattr(self, '_calls', 0) - before, ablation_solver_seconds=time.monotonic() - start, ablation_matched_initialization=True, ablation_posterior_information_added_once=True, ablation_common_partition_root=self.prior['counterfactual_partition_root'])
        return (pred, diag)

def setup():
    torch.set_num_threads(1)
    import trajectory_cross_region as module
    module.CrossRegionPath = M2ControlPath
    capture = BASE / 'runs/direction_v80/trajectory_factor_capture_v1'
    features = BASE / 'runs/direction_v80/trajectory_geometry_readout_full_v1'
    parent = BASE / 'runs/direction_v80/trajectory_temporal_radio_v1'
    source = HERE / 'M3_module_full/training/head_prior.json'
    if not source.exists():
        raise ValueError('M3 common source-training/export dependency not ready')
    prior = json.loads(source.read_text())
    prior.update(RSSI_cross_region_objective=True, counterfactual_partition_root=str(parent), counterfactual_region_width_m=2.0, counterfactual_predictive_strength=1.0, cache_solver_groups=True, save_all_calibration_candidates=False, path_solver_maxiter=300)
    return (capture, features, parent, prior)

def worker_setup():
    setup()

def proof():
    from trajectory_axial_motion import feature_speeds
    from support_label_barrier import deny_label_reads
    capture, features, parent, prior = setup()
    s = json.loads((capture / 'segments.json').read_text())['source_train'][0]
    e = feature_speeds(features)['source_train', s['trajectory_id']]
    source = capture / 'source_train' / s['trajectory_id']
    saved = json.loads((parent / 'source_train' / s['trajectory_id'] / 'predictions.json').read_text())
    xy = np.array([r['xy_m'] for r in saved])
    solvers = {}
    values = {}
    times = {}
    objective_times = {}
    with deny_label_reads() as audit:
        for policy in ['cross_region', 'global_marginal', 'self_predictive']:
            p = dict(prior, ablation_M2_score=policy)
            solver = M2ControlPath(source, s, p, (e['speed'], e['sigma'], np.ones(len(xy), bool)), False, xy, force_context=e)
            q = np.stack([np.interp(solver.knots, np.r_[0.0, solver.t], np.vstack([solver.initial, xy])[:, j]) for j in range(2)], -1)
            q[0] = solver.initial
            start = time.monotonic()
            value, gradient, _ = solver.radio(q, 0)
            times[policy] = time.monotonic() - start
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise ValueError('Nonfinite actual M2 score or gradient')
            solvers[policy] = (solver, q)
            values[policy] = dict(score=value, gradient_norm=float(np.linalg.norm(gradient)))
            vector = np.r_[q[1:].ravel(), np.zeros(4), 0.0]
            durations = []
            solver.axial_objective(vector, 0)
            for _ in range(3):
                start = time.monotonic()
                solver.axial_objective(vector, 0)
                durations.append(time.monotonic() - start)
            objective_times[policy] = float(np.median(durations))
        solver, q = solvers['self_predictive']
        model = solver.observations
        mean, B = model.geometry(solver.raw_spline @ q, solver.radio_field)
        z, A, ld = model.whiten(mean, B, solver.radio_blocks)
        value, gz, gA, pa = self_predictive(z, A, solver.device_prior, ld)
        _, native_mean, _, _, _ = mixture_posterior(z, A, solver.device_prior, ld)
        mean_difference = float(np.max(abs(pa['posterior_mean'] - native_mean)))
        precision_difference = float(np.max(abs(pa['posterior_precision'] - solver.device_prior['precision'] - (A.T @ A)[None])))
        if mean_difference > 1e-07 or precision_difference > 1e-07:
            raise ValueError('A2 posterior differs from exactly one evidence assimilation')
        rng = np.random.RandomState(903)
        direction = rng.normal(size=q.shape)
        direction /= np.linalg.norm(direction)
        step = 1e-05
        actual = float((solver.radio(q + step * direction, 0)[0] - solver.radio(q - step * direction, 0)[0]) / (2 * step))
        analytic = float((solver.radio(q, 0)[1] * direction).sum())
        gradient_error = abs(actual - analytic)
        if gradient_error > 0.002 * (1 + abs(actual)):
            raise ValueError('A2 predictive objective pullback gradient differs')
    report = dict(status='PASS_ACTUAL_SOURCE_M2_INTERVENTIONS', source_trajectory=s['trajectory_id'], scores=values, posterior_mean_difference=mean_difference, posterior_single_assimilation_precision_difference=precision_difference, self_score_directional_gradient_difference=gradient_error, blocked_label_attempts=audit['blocked_label_attempts'], raw_CS_and_RSSI_factors_unchanged=True, A1_A2_equivalent=False, distinction='A1 integrates same global device model; A2 evaluates posterior predictive under that once-fitted posterior; no artificial precision multiplication', scope='Full conditional objective uses fixed sensor-parent spatial folds, not full independently replayed path/context for every fold', single_radio_call_seconds=times, source_median_whole_objective_seconds=objective_times, A1_source_benchmarked_equal_compute_iteration_limit=max(300, int(math.ceil(300 * objective_times['cross_region'] / objective_times['global_marginal']))))
    write(HERE / 'M2_actual_path_proof.json', report)
    print(json.dumps(report), flush=True)
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proof-only', action='store_true')
    a = ap.parse_args()
    if a.proof_only:
        proof()
        return
    with (BASE / 'research_gpu0_ablation.lock').open('a+') as lock:
        print('QUEUED_M2_GPU0_ABLATIONS', os.getpid(), flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        import trajectory_axial_motion as runtime
        from trajectory_joint_marginal import score
        evidence = json.loads((HERE / 'M2_actual_path_proof.json').read_text()) if (HERE / 'M2_actual_path_proof.json').exists() else proof()
        if evidence['status'] != 'PASS_ACTUAL_SOURCE_M2_INTERVENTIONS':
            raise ValueError('M2 functional proof missing')
        capture, features, parent, prior = setup()
        results = []
        runtime.ProcessPoolExecutor = functools.partial(concurrent.futures.ProcessPoolExecutor, mp_context=multiprocessing.get_context('spawn'), initializer=worker_setup)
        equal = evidence['A1_source_benchmarked_equal_compute_iteration_limit']
        arms = [('M2_CF_runtime', 'cross_region', 300), ('M2_A1_global_equal_compute', 'global_marginal', equal), ('M2_A2_self_runtime', 'self_predictive', 300), ('M2_A1_global_extended', 'global_marginal', max(1000, 2 * equal))]
        for name, policy, iterations in arms:
            folder = HERE / name
            folder.mkdir(exist_ok=True)
            p = dict(prior, ablation_M2_score=policy, path_solver_maxiter=iterations)
            pp = folder / 'prior.json'
            if pp.exists():
                frozen = json.loads(pp.read_text())
                if frozen != p:
                    raise ValueError('Preserve the frozen arm protocol on resume')
            else:
                write(pp, p)
            args = SimpleNamespace(base=BASE, capture=capture, features=features, learned_path=parent, alternative=parent, prior=pp, output=folder, workers=24, source_fold='all', cohorts=['source_train', 'original_fixed20', 'AP_movement'])
            if not (folder / 'prediction_complete.json').exists():
                runtime.infer(args)
            if not (folder / 'comparison.json').exists():
                args.cohorts = ['original_fixed20', 'AP_movement']
                score(args)
            results.append(dict(arm=name, policy=policy, iteration_upper_bound=iterations, comparison=str(folder / 'comparison.json')))
        write(HERE / 'M2_runtime_interventions_completion.json', dict(status='COMPLETED', results=results, equal_compute_note='Identical starts and early-stop rules; A1 allowance is scaled from source-only median whole-objective timings to the300iteration Full budget. Actual objective evaluations and CPU seconds are recorded; extendedA1 has at least1000iterations and twice the matched allowance. This is source-benchmarked approximate compute matching, not identical consumed target wall time.', all_requested_M2_evidence_complete=False, pending='Matched-budget same-region comparison and ambiguity/candidate coverage/held-region forecast linkage diagnostics'))
if __name__ == '__main__':
    main()
