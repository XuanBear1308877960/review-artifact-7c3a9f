import argparse
import concurrent.futures
import fcntl
import functools
import json
import multiprocessing
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
BASE = Path('.')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from trajectory_joint_calibration import field as ORIGINAL_FIELD
from trajectory_joint_marginal import initial_prior as ORIGINAL_PRIOR
from run_m2_pair import setup

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

class BandAffineField:

    def __init__(self, parent):
        self.parent = parent
        for name in ['xy', 'mask', 'values', 'centres', 'ap', 'band', 'radius2']:
            setattr(self, name, getattr(parent, name))
        _, self.band_index = np.unique(self.band, axis=0, return_inverse=True)
        if self.band_index.max() > 1:
            raise ValueError('Six-D protocol allows two measured WiFi bands')

    def affine(self, mean, dm, ap):
        band = self.band_index[ap]
        B = np.zeros(mean.shape + (6,))
        db = None if dm is None else np.zeros(mean.shape + (6, 2))
        for j in range(2):
            use = band == j
            B[..., 2 * j] = use
            B[..., 2 * j + 1] = use * mean / 50
            if dm is not None:
                db[..., 2 * j + 1, :] = use[..., None] * dm / 50
        return (B, db)

    def ap_query(self, points, context, ap, derivative=False):
        if derivative:
            mean, dm = self.parent.source_ap(points, ap, True)
        else:
            mean = self.parent.source_ap(points, ap)
            dm = None
        B, db = self.affine(mean, dm, ap)
        return (mean, B, dm, db) if derivative else (mean, B)

    def __call__(self, points, context, derivative=False):
        ap = np.broadcast_to(np.arange(4), (len(points), 4))
        xy = np.broadcast_to(points[:, None], (len(points), 4, 2)).reshape(-1, 2)
        ctx = np.broadcast_to(context[:, None], (len(points), 4, 32)).reshape(-1, 32)
        output = self.ap_query(xy, ctx, ap.ravel(), derivative)
        return tuple((a.reshape((len(points), 4) + a.shape[1:]) for a in output))

    def source_ap(self, *args, **kwargs):
        return self.parent.source_ap(*args, **kwargs)

    def variance_ap(self, *args, **kwargs):
        return self.parent.variance_ap(*args, **kwargs)

def selected_field(base, prior=None):
    original = ORIGINAL_FIELD(base, prior)
    return BandAffineField(original) if prior and prior.get('a5_band_affine_response', False) else original

def projection(pre):
    base = ORIGINAL_FIELD(BASE)
    affine = BandAffineField(base)
    rp, ap = np.where(base.mask > 0.5)
    points = base.xy[rp]
    context = np.broadcast_to(pre['motion'][4], (len(points), 32))
    _, old = base.ap_query(points, context, ap)
    _, new = affine.ap_query(points, context, ap)
    weight = 1 / np.bincount(ap, minlength=4)[ap]
    normal = new.T @ (weight[:, None] * new) + np.eye(6) * 1e-06
    T = np.linalg.solve(normal, new.T @ (weight[:, None] * old))
    active = np.any(new != 0, axis=0)
    return (T, active, old, new)

def affine_initial_prior(pre, uncertainty_scale=1.0):
    p = ORIGINAL_PRIOR(pre, uncertainty_scale)
    T, active, _, _ = projection(pre)
    mu = p['mean'] @ T.T
    cov = np.einsum('id,kde,je->kij', T, p['covariance'], T)
    cov += np.diag(np.where(active, 0.0001, 1.0))[None]
    precision = np.linalg.inv(cov)
    return dict(mean=mu, covariance=cov, precision=precision, vector=np.einsum('kde,ke->kd', precision, mu), logdet=np.linalg.slogdet(cov)[1], log_weight=p['log_weight'].copy())

def worker_setup():
    setup()
    import trajectory_temporal_radio as temporal
    import trajectory_joint_marginal as marginal
    import trajectory_robust_initial as initial
    temporal.field = selected_field
    temporal.initial_prior = affine_initial_prior
    marginal.field = selected_field
    initial.initial_prior = affine_initial_prior

def proof():
    from support_label_barrier import deny_label_reads
    capture, _, _, _ = setup()
    s = json.loads((capture / 'segments.json').read_text())['source_train'][0]
    with deny_label_reads() as audit:
        with np.load(capture / 'source_train' / s['trajectory_id'] / 'PRE_forecasts.npz', allow_pickle=False) as z:
            pre = {k: z[k].copy() for k in z.files}
        base = ORIGINAL_FIELD(BASE)
        affine = BandAffineField(base)
        points = base.xy[::23]
        ctx = np.broadcast_to(pre['motion'][4], (len(points), 32))
        old = base(points, ctx, True)
        new = affine(points, ctx, True)
        if not np.array_equal(old[0], new[0]) and np.max(abs(old[0] - new[0])) > 1e-12:
            raise ValueError('A5 changed OPPO mean')
        T, active, O, N = projection(pre)
        p = ORIGINAL_PRIOR(pre)
        q = affine_initial_prior(pre)
        if np.linalg.eigvalsh(q['covariance']).min() <= 0:
            raise ValueError('A5 projected prior is singular')
        forward = np.einsum('nad,kd->kna', new[1], q['mean'])
        d = np.random.RandomState(905).normal(size=points.shape)
        d /= np.linalg.norm(d)
        h = 1e-05
        numeric = (affine(points + h * d, ctx)[1] - affine(points - h * d, ctx)[1]) / (2 * h)
        error = float(abs(numeric - np.einsum('nadz,nz->nad', new[3], d)).max())
        if error > 1e-05:
            raise ValueError('A5 affine derivative differs')
    report = dict(status='PASS_SOURCE_A5_FIELD_AND_PROJECTED_PRIOR', OPPO_mean_max_difference=float(abs(old[0] - new[0]).max()), affine_derivative_difference=error, active_device_parameters=int(active.sum()), downstream_latent_dimension=6, initial_mapping_projection_RMSE_db=float(np.sqrt(np.square(O @ p['mean'].T - N @ q['mean'].T).mean())), initial_prior_projection_uses='Known OPPO grid plus immutable initial-five target context/posterior only', blocked_label_attempts=audit['blocked_label_attempts'])
    write(HERE / 'A5_affine_proof.json', report)
    print(json.dumps(report), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proof-only', action='store_true')
    a = ap.parse_args()
    if a.proof_only:
        proof()
        return
    with (BASE / 'research_gpu0_ablation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        proof()
        worker_setup()
        import trajectory_temporal_radio as temporal
        import trajectory_axial_motion as runtime
        from trajectory_joint_marginal import score
        capture, features, parent, prior = setup()
        out = HERE / 'M1_A5_band_affine'
        out.mkdir(exist_ok=True)
        prior.update(a5_band_affine_response=True, ablation_M2_score='cross_region', path_solver_maxiter=300)
        initial = out / 'before_noise_fit.json'
        if not initial.exists():
            write(initial, prior)
        pp = out / 'prior.json'
        if not pp.exists():
            fit = SimpleNamespace(base=BASE, capture=capture, queries=Path(prior['temporal_RSSI_query_root']), prior=initial, output=pp, freeze_field_context=False, lock_spatial_basis=False)
            temporal.fit(fit)
        runtime.ProcessPoolExecutor = functools.partial(concurrent.futures.ProcessPoolExecutor, mp_context=multiprocessing.get_context('spawn'), initializer=worker_setup)
        args = SimpleNamespace(base=BASE, capture=capture, features=features, learned_path=parent, alternative=parent, prior=pp, output=out, workers=24, source_fold='all', cohorts=['source_train', 'original_fixed20', 'AP_movement'])
        if not (out / 'prediction_complete.json').exists():
            runtime.infer(args)
        if not (out / 'comparison.json').exists():
            args.cohorts = ['original_fixed20', 'AP_movement']
            score(args)
        write(HERE / 'A5_affine_completion.json', dict(status='COMPLETED_AFFINE_FIELD_RUNTIME_CONTROL', full='M2_CF_runtime', ablation=str(out), OPPO_mean_preserved=True, same_M2_objective_and_motion_CS_backend=True, initial_prior_reexpressed_in_affine_coordinates=True, source_temporal_noise_fit_budget='Same source54train/held split,13noise parameters,100L-BFGS iteration upper limit', M3_scope='Same frozen calibrated spatial potential in both runtime arms; this isolates the physical field response with unchanged downstream potential, not end-to-end retrained M3 features', target_GT_in_fitting=False))
if __name__ == '__main__':
    main()
