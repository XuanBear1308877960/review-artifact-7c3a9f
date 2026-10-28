import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np
from scipy.linalg import block_diag, cholesky, cho_solve
from scipy.special import logsumexp, softmax
import torch
from canonical_replay import BASE, HERE, ROOT, write
sys.path.insert(0, str(BASE))
from trajectory_temporal_radio import TemporalJointPath, ObservationModel
from trajectory_axial_motion import feature_speeds
from trajectory_joint_marginal import posterior, initial_prior
CAPTURE = Path(os.environ.get('YOMO_CANONICAL_CAPTURE_ROOT', str(BASE / 'runs/direction_v80/trajectory_factor_capture_v1')))
PARENTS = [BASE / 'runs/direction_v80/trajectory_incumbent_M2_selected_v1', BASE / 'runs/direction_v80/motion_family_v1/complete_pool']
_FEATURES = None
_PRIORS = {}

def priors(parent):
    if parent not in _PRIORS:
        p = json.loads((PARENTS[parent] / 'prediction_complete.json').read_text())['source_prior']
        p = dict(p)
        p.pop('additional_parent_solver_root', None)
        p.pop('solver_polish_parent_root', None)
        p.update(save_all_calibration_candidates=False, path_solver_maxiter=300, cache_solver_groups=False)
        if os.environ.get('YOMO_CANONICAL_NEURAL_MOTION_PRIOR') == '1':
            p['canonical_neural_motion_prior'] = 'Frozen77187parameter RTS-shape-GRU; recomputed from this context, no full-data query feature cache'
        if os.environ.get('YOMO_CANONICAL_DIRECTION_SEED') == '1':
            p['canonical_direction_seed'] = 'Frozen source directional GRU/Viterbi, regenerated separately from sanitized context observations'
        if (ROOT / 'control.json').exists():
            control = json.loads((ROOT / 'control.json').read_text())
            if control['case'] == 'A4_no_feedback':
                p.update(canonical_A4_no_WiFi_to_CS=True, canonical_control='A4_no_feedback')
            elif control['case'] == 'A5_band_affine':
                p.update(canonical_A5_band_affine=True, canonical_control='A5_band_affine')
            elif control['case'] == 'S1_no_Jacobian':
                from canonical_replay import FULL_ROOT
                model = FULL_ROOT / 'active_basis_structure/without_Jacobian/training/basis_residual.npz'
                if not (FULL_ROOT / 'active_basis_structure/training_pair_complete.json').exists():
                    raise ValueError('Complete the equal-budget active-basis source pair first')
                p.update(calibration_basis_residual_model=str(model), canonical_control='S1_no_Jacobian', canonical_M1_Jacobian_penalty_weight=0.0)
            elif control['case'] == 'A2_self_context_CV':
                p.update(canonical_context_self_predictive=True, canonical_control='A2_self_context_CV')
            else:
                raise ValueError('Unknown canonical control')
        _PRIORS[parent] = p
    return dict(_PRIORS[parent])

def evidence(cohort, tid):
    global _FEATURES
    if _FEATURES is None:
        _FEATURES = feature_speeds(BASE / 'runs/direction_v80/trajectory_geometry_readout_full_v1')
    return _FEATURES[cohort, tid]

class ContextFitPath(TemporalJointPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from canonical_field_controls import configure as configure_field
        configure_field(self)
        from canonical_CS_feedback import configure
        configure(self)

    def calibrate_solution(self, predictions, diagnostic):
        from canonical_CS_feedback import diagnostics
        from canonical_field_controls import diagnostics as field_diagnostics
        diagnostic.update(diagnostics(self))
        diagnostic.update(field_diagnostics(self))
        return (predictions, diagnostic)

class ContextSelfFitPath(ContextFitPath):

    def radio(self, q, mode):
        from scipy.linalg import solve_triangular
        from run_m2_pair import self_predictive
        from trajectory_joint_marginal import mixture_posterior
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
        _, mu, P, _, weights = mixture_posterior(z, A, self.device_prior, ld)
        return (loss, model.pullback(gm, gb, dm, db, self.raw_spline), (mu, P, weights))

def solver_for(spec, parent):
    folder = Path(spec['folder'])
    s = spec['segment']
    cohort = spec['cohort']
    e = evidence(cohort, s['trajectory_id'])
    xy = np.array([r['xy_m'] for r in json.loads((folder / 'predictions.json').read_text())])
    p = priors(parent)
    cls = ContextSelfFitPath if p.get('canonical_context_self_predictive', False) else ContextFitPath
    learned = xy
    if os.environ.get('YOMO_CANONICAL_NEURAL_MOTION_PRIOR') == '1':
        path = folder.parent.parent / 'source_motion_prior' / cohort / s['trajectory_id'] / 'predictions.json'
        learned = np.array([r['xy_m'] for r in json.loads(path.read_text())])
        name = 'direction_motion_seed' if os.environ.get('YOMO_CANONICAL_DIRECTION_SEED') == '1' else 'rts_mean'
        alternative = folder.parent.parent / name / cohort / s['trajectory_id'] / 'predictions.json'
        xy = np.array([r['xy_m'] for r in json.loads(alternative.read_text())])
    solver = cls(folder, s, p, (e['speed'], e['sigma'], np.ones(len(xy), bool)), False, learned, force_context=e)
    return (solver, xy)

def cache_path(spec, parent, group):
    return ROOT / 'context_fits' / ('parent_%d' % parent) / spec['cohort'] / spec['segment']['trajectory_id'] / ('fold_%03d' % spec['fold']) / ('group_%d.pkl' % group)

def fit_one(task):
    spec, parent, group = task
    path = cache_path(spec, parent, group)
    start = time.monotonic()
    torch.set_num_threads(1)
    replay = json.loads((Path(spec['folder']) / 'completion.json').read_text())
    signature = hashlib.sha256(json.dumps(dict(prior=priors(parent), replay_state=replay['state_sha256'], group=group), sort_keys=True).encode()).hexdigest()
    if path.exists():
        with path.open('rb') as f:
            r = pickle.load(f)
        if r['signature'] != signature:
            raise ValueError('Context fit cache signature changed')
        return r['report']
    from support_label_barrier import deny_label_reads
    with deny_label_reads() as audit:
        solver, xy = solver_for(spec, parent)
        pred, diag = solver.solve_axis(xy, [group, group + 1])
        vector = solver.last_selected_vector.copy()
        q, _ = solver.unpack(vector[:-1])
        model = solver.observations
        mean, B = model.geometry(solver.raw_spline @ q, solver.radio_field)
        z, A, ld = model.whiten(mean, B, solver.radio_blocks)
        base = solver.device_prior
        components = [posterior(z, A, base, k, ld) for k in range(len(base['mean']))]
        cost = np.array([v[0] for v in components])
        logw = base['log_weight'] - cost
        logw -= logsumexp(logw)
        precision = base['precision'] + (A.T @ A)[None]
        mu = np.array([v[1] for v in components])
        P = np.array([v[2] for v in components])
        fitted = dict(mean=mu, covariance=P, precision=precision, vector=np.einsum('kde,ke->kd', precision, mu), logdet=np.linalg.slogdet(P)[1], log_weight=logw)
        if not all((np.isfinite(x).all() for x in [q, mu, P])):
            raise ValueError('Nonfinite context path/posterior')
        report = dict(cohort=spec['cohort'], trajectory_id=spec['segment']['trajectory_id'], fold=spec['fold'], parent=parent, group=group, objective=diag['objective'], iterations=diag['iterations'], stopping_reason=diag['stopping_reason'], seconds=time.monotonic() - start, query_AP_observations=spec['query_AP_observations'], support_AP_observations=len(z), blocked_label_attempts=audit['blocked_label_attempts'])
        if audit['blocked_label_attempts']:
            raise ValueError('Context fit attempted reference reads')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.pending').open('wb') as f:
        pickle.dump(dict(signature=signature, report=report, diagnostic=diag, vector=vector, knots=solver.knots.copy(), q=q, xy=np.array([r['xy_m'] for r in pred]), posterior=fitted), f, protocol=4)
    path.with_suffix('.pending').replace(path)
    return report

def flat_indices(mask):
    return np.flatnonzero(mask.T.ravel())

def assemble_one(task):
    spec, parent = task
    destination = ROOT / 'conditional_packages' / ('parent_%d' % parent) / spec['cohort'] / spec['segment']['trajectory_id'] / ('fold_%03d.pkl' % spec['fold'])
    if destination.exists():
        return dict(path=str(destination), already_complete=True)
    from support_label_barrier import deny_label_reads
    with deny_label_reads() as audit:
        records = []
        for group in [0, 2, 4, 6]:
            with cache_path(spec, parent, group).open('rb') as f:
                records.append(pickle.load(f))
        records.sort(key=lambda r: r['report']['objective'])
        unique = []
        for record in records:
            if all((np.sqrt(np.square(record['xy'] - other['xy']).sum(-1).mean()) > 0.05 for other in unique)):
                unique.append(record)
        weights = softmax(-np.array([r['report']['objective'] for r in unique]))
        solver, _ = solver_for(spec, parent)
        model = solver.observations
        N = model.N
        with np.load(Path(spec['folder']) / 'masks.npz', allow_pickle=False) as z:
            query = z['query'].copy()
            support = z['support'].copy()
        if not np.array_equal(support, model.mask):
            raise ValueError('Regenerated native mask differs from registered support')
        qi = flat_indices(query)
        si = flat_indices(support)
        scan = qi % N
        ap = qi // N
        cov, _, _ = model.noise(np.asarray(solver.prior['temporal_RSSI_parameters']))
        R = block_diag(*cov)
        if len(si):
            Ls = cholesky(R[np.ix_(si, si)], lower=True)
            K = cho_solve((Ls, True), R[np.ix_(si, qi)]).T
            conditional = R[np.ix_(qi, qi)] - K @ R[np.ix_(si, qi)]
        else:
            K = np.zeros((len(qi), 0))
            conditional = R[np.ix_(qi, qi)]
        Lq = cholesky(0.5 * (conditional + conditional.T), lower=True)
        groups = []
        observed_support = model.pre['rssi'].T.ravel()[si]
        for weight, record in zip(weights, unique):
            mean, B = model.geometry(solver.raw_spline @ record['q'], solver.radio_field)
            means = mean.T.ravel()[si]
            basis = B.transpose(1, 0, 2).reshape(4 * N, 6)[si]
            groups.append(dict(log_weight=float(np.log(weight)), posterior=record['posterior'], noise_offset=K @ (observed_support - means), basis_offset=K @ basis, context_path_xy=record['xy'], context_q=record['q'], objective=record['report']['objective']))
        full_W = np.zeros((len(qi), len(model.raw_times)))
        for j, (i, a) in enumerate(zip(scan, ap)):
            full_W[j, model.bounds[a]:model.bounds[a + 1]] = model.S[a][i]
        raw = np.flatnonzero(full_W.any(0))
        W = full_W[:, raw]
        raw_spline = solver.raw_spline[raw]
        raw_context = model.raw_context()[raw]
        raw_AP = model.ap[raw]
        for a in range(4):
            qraw = model.S[a][query[:, a]].sum(0) > 0
            sraw = model.S[a][support[:, a]].sum(0) > 0
            if (qraw & sraw).any():
                raise ValueError('Query/support share raw median contributors')
        reference_q = sum((weight * record['q'] for weight, record in zip(weights, unique)))
        reference_xy = solver.B @ reference_q
        atom_row, atom_raw = np.nonzero(W)
        atom_shift = (raw_spline @ reference_q)[atom_raw] - reference_xy[scan[atom_row]]
        original = CAPTURE / spec['cohort'] / spec['segment']['trajectory_id']
        with np.load(original / 'PRE_forecasts.npz', allow_pickle=False) as z:
            if not np.array_equal(z['observation_ids'], model.pre['observation_ids']):
                raise ValueError('Query IDs differ')
            y = z['rssi'].T.ravel()[qi].astype(float)
        signature = hashlib.sha256()
        for group in groups:
            for k in sorted(group['posterior']):
                signature.update(np.ascontiguousarray(group['posterior'][k]).tobytes())
            signature.update(group['noise_offset'].tobytes())
            signature.update(group['basis_offset'].tobytes())
        for array in [Lq, raw_context, raw_spline, W]:
            signature.update(np.ascontiguousarray(array).tobytes())
        package = dict(cohort=spec['cohort'], trajectory_id=spec['segment']['trajectory_id'], fold=spec['fold'], region=spec['region'], parent=parent, query_scan=scan, query_AP=ap, query_ids=model.pre['observation_ids'][scan], query_y=y, query_mask=query, support_mask=support, query_noise_cholesky=Lq, query_noise_logdet=float(2 * np.log(np.diag(Lq)).sum()), groups=groups, query_raw_indices=raw, query_raw_times=model.raw_times[raw], raw_spline=raw_spline, raw_context=raw_context, raw_AP=raw_AP, W=W, atom_row=atom_row, atom_raw=atom_raw, atom_weight=W[atom_row, atom_raw], atom_shift=atom_shift, field_motion=model.pre['motion'], reference_xy=reference_xy, reference_q=reference_q, knots=solver.knots.copy(), M2_frozen_sha256=signature.hexdigest(), context_replayed_without_query=True, raw_overlap=0, context_group_count_before_dedup=4, context_group_count_after_dedup=len(unique), context_GEOMETRY_scope='Energy-weighted discrete context MAP alternatives, not continuous-path evidence integral')
        if audit['blocked_label_attempts']:
            raise ValueError('Conditional assembly attempted reference reads')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix('.pending').open('wb') as f:
        pickle.dump(package, f, protocol=4)
    destination.with_suffix('.pending').replace(destination)
    return dict(path=str(destination), cohort=spec['cohort'], trajectory_id=spec['segment']['trajectory_id'], fold=spec['fold'], parent=parent, query_AP_observations=len(qi), support_AP_observations=len(si), context_groups=len(unique), M2_frozen_sha256=signature.hexdigest())

def execute(function, tasks, workers):
    reports = []
    start = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(function, t): t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            report = future.result()
            reports.append(report)
            if len(reports) % 32 == 0 or len(reports) == len(tasks):
                print(json.dumps(dict(stage=function.__name__, completed=len(reports), total=len(tasks), seconds=time.monotonic() - start)), flush=True)
    return reports

def main():
    import contextlib
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-prototype', action='store_true')
    ap.add_argument('--workers', type=int, default=96)
    ap.add_argument('--lock-owned', action='store_true')
    ap.add_argument('--parents', type=int, nargs='+', choices=[0, 1], default=[0, 1])
    ap.add_argument('--cohorts', nargs='+', default=['source_train', 'original_fixed20', 'AP_movement'])
    a = ap.parse_args()
    with contextlib.nullcontext() if a.lock_owned else (BASE / 'research_gpu0_ablation.lock').open('a+') as lock:
        if not a.lock_owned:
            fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = ROOT / ('source_prototype_fold_manifest.json' if a.source_prototype else 'fold_manifest.json')
        specs = json.loads(manifest.read_text())['specs']
        if not a.source_prototype and (not (ROOT / 'replays_complete.json').exists()):
            raise ValueError('Wait for all regenerated context replays')
        if a.source_prototype:
            specs = specs[:1]
        specs = [s for s in specs if s['cohort'] in a.cohorts]
        tasks = [(spec, parent, group) for spec in specs for parent in a.parents for group in [0, 2, 4, 6]]
        fits = execute(fit_one, tasks, min(a.workers, len(tasks)))
        assembled = execute(assemble_one, [(s, p) for s in specs for p in a.parents], min(24, len(specs) * len(a.parents)))
        write(ROOT / ('source_prototype_context_fit_complete.json' if a.source_prototype else 'context_fit_complete.json'), dict(status='COMPLETED_CONTEXT_ONLY_PATH_FITS_AND_QUERY_PACKAGES', groups=len(fits), packages=assembled, source_only_prototype=a.source_prototype, CPU_workers=a.workers, parents=a.parents, cohorts=a.cohorts, query_RSSI_entered_context_fit=False, source_best_physics_reused=True))
if __name__ == '__main__':
    main()
