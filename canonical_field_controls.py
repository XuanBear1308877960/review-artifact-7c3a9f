import numpy as np
from canonical_replay import BASE
from trajectory_joint_calibration import field
from run_a5_affine import BandAffineField

def selected_field(base, prior):
    original = field(base, prior)
    return BandAffineField(original) if prior.get('canonical_A5_band_affine', False) else original

def project_prior(old, pre, original, affine):
    rp, ap = np.where(original.mask > 0.5)
    points = original.xy[rp]
    context = np.broadcast_to(pre['motion'][4], (len(points), 32))
    _, full_B = original.ap_query(points, context, ap)
    _, affine_B = affine.ap_query(points, context, ap)
    weights = 1 / np.bincount(ap, minlength=4)[ap]
    T = np.linalg.solve(affine_B.T @ (weights[:, None] * affine_B) + np.eye(6) * 1e-06, affine_B.T @ (weights[:, None] * full_B))
    active = np.any(affine_B != 0, axis=0)
    mu = old['mean'] @ T.T
    covariance = np.einsum('id,kde,je->kij', T, old['covariance'], T) + np.diag(np.where(active, 0.0001, 1.0))[None]
    precision = np.linalg.inv(covariance)
    new = dict(mean=mu, covariance=covariance, precision=precision, vector=np.einsum('kde,ke->kd', precision, mu), logdet=np.linalg.slogdet(covariance)[1], log_weight=old['log_weight'].copy())
    diagnostic = dict(active_parameters=int(active.sum()), projection_RMSE_db=float(np.sqrt(np.square(full_B @ old['mean'].T - affine_B @ new['mean'].T).mean())), projection_uses_full_selected_1732_basis=True, projection_RP_rows=len(points), target_reference_used=False)
    return (new, diagnostic)

def configure(solver):
    if not solver.prior.get('canonical_A5_band_affine', False):
        return
    original = solver.radio_field
    affine = BandAffineField(original)
    solver.device_prior, solver.canonical_A5_diagnostic = project_prior(solver.device_prior, solver.pre, original, affine)
    solver.radio_field = affine

def diagnostics(solver):
    if not solver.prior.get('canonical_A5_band_affine', False):
        return {}
    return dict(canonical_A5_band_affine=True, canonical_A5_selected_OPPO_mean_preserved=True, canonical_A5_projected_initial_prior=solver.canonical_A5_diagnostic)
