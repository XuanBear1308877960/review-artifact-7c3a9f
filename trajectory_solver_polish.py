import math
import numpy as np
from scipy.optimize import minimize

def polish_axis(solver, modes):
    saved = solver.polish_parent_state
    state = saved['optimization_state']
    x = state['vector'].copy()
    if not solver.prior.get('joint_marginal_initial_mixture'):
        raise ValueError('Polishing currently requires the same eight-component joint marginal model')
    if not np.allclose(state['knots_s'], solver.knots, atol=1e-10):
        raise ValueError('Polish spline convention differs')
    mode = int(saved['diagnostic']['calibration_mode'])
    before, g0 = solver.axial_objective(x, mode)
    difference = float(abs(before - saved['diagnostic']['objective']))
    if difference > 1e-08 * (1 + abs(before)):
        raise ValueError('Polish model differs from the saved objective')
    result = minimize(solver.axial_objective, x, args=(mode,), jac=True, method='L-BFGS-B', options=dict(maxiter=int(solver.prior.get('solver_polish_maxiter', 1000)), ftol=float(solver.prior.get('solver_polish_ftol', 1e-12)), gtol=float(solver.prior.get('solver_polish_gtol', 2e-05)), maxls=60))
    use_refined = bool(np.isfinite(result.fun) and result.fun <= before)
    selected = result.x if use_refined else x
    value, g = solver.axial_objective(selected, mode)
    q, _ = solver.unpack(selected[:-1])
    q0, _ = solver.unpack(x[:-1])
    xy = solver.B @ q
    change = np.linalg.norm(solver.B @ (q - q0), axis=-1)
    solver.last_selected_vector = selected.copy()
    predictions = [dict(p, xy_m=point.tolist(), inference_timing='OFFLINE_SAME_MODEL_REFINED_SOLVER', trajectory_calibration_mode=mode, later_position_GT_used=False) for p, point in zip(solver.predictions, xy)]
    diagnostic = dict(converged=bool(result.success), iterations=int(result.nit), objective=float(value), axis_degrees=float(np.rad2deg(selected[-1]) % 180), branch_objectives=[float(before), float(result.fun)], stopping_reason=str(result.message), optimization_starts=1, branch_stopping_reasons=[str(result.message)], calibration_mode=mode, later_GT_used=False, source_scene_motion_prior=True, phone_heading_used=False, initial_device_mixture_marginalized=True, optimization_native_seed_modes=modes, solver_polish_same_model=True, solver_polish_parent_objective=float(before), solver_polish_parent_objective_difference=difference, solver_polish_refined_selected=use_refined, solver_polish_original_position_gradient_inf=float(abs(g0[:-5]).max()), solver_polish_position_gradient_inf=float(abs(g[:-5]).max()), solver_polish_original_nuisance_gradient_inf=float(abs(g0[-5:]).max()), solver_polish_nuisance_gradient_inf=float(abs(g[-5:]).max()), solver_polish_mean_prediction_change_m=float(change.mean()), solver_polish_max_prediction_change_m=float(change.max()), solver_polish_fixed_initial_position=True, solver_polish_parameters_or_measurements_changed=False, solver_polish_ftol=solver.prior.get('solver_polish_ftol', 1e-12), solver_polish_gtol=solver.prior.get('solver_polish_gtol', 2e-05))
    return (predictions, diagnostic)
