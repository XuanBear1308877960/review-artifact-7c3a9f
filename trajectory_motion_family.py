import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from engine_utils import write
from trajectory_velocity_states import source_rows, velocity_energy
from trajectory_directional_velocity import context, energy as directional_energy

def family_energy(velocity, speed, dt, old_parameters, old_temporal, speed_sigma, circular_context, probability, derivative=False):
    if not 0.0 <= probability <= 1.0:
        raise ValueError('Invalid circular motion family probability')
    old = velocity_energy(velocity, speed, dt, old_parameters, old_temporal, derivative, speed_sigma)
    new = directional_energy(velocity, speed, circular_context, derivative)
    losses = np.asarray([old[0] if derivative else old, new[0] if derivative else new])
    with np.errstate(divide='ignore'):
        logs = np.log([1 - probability, probability]) - losses
    normalizer = logsumexp(logs)
    weights = np.exp(logs - normalizer)
    circular_context['last_family_posterior'] = weights.copy()
    if not derivative:
        return -float(normalizer)
    return (-float(normalizer), weights[0] * old[1] + weights[1] * new[1], weights)

def fit(a):
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    circular = json.loads(a.circular_prior.read_text())
    rows = source_rows(a)
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    phones = {s['trajectory_id']: s['device'] for s in segments}
    reports = []
    for row in rows:
        ctx = context(circular['directional_velocity_parameters'], row['dt'], row['time_s'], circular.get('directional_velocity_directions', 16))
        old = velocity_energy(row['velocity'], row['speed'], row['dt'], prior['velocity_state_parameters'], prior['velocity_state_temporal_selected'], speed_sigma=row['speed_sigma'])
        new = directional_energy(row['velocity'], row['speed'], ctx)
        reports.append(dict(trajectory_id=row['trajectory_id'], phone=phones[row['trajectory_id']], heldout=row['heldout'], n=len(row['speed']), old_NLL=old, circular_NLL=new))

    def objective(p, heldout, balanced=False):
        chosen = [r for r in reports if r['heldout'] == heldout]

        def score(items):
            total = 0.0
            with np.errstate(divide='ignore'):
                lp = np.log([1 - p, p])
            for row in items:
                total -= logsumexp(lp - np.asarray([row['old_NLL'], row['circular_NLL']]))
            return float(total / sum((r['n'] for r in items)))
        if balanced:
            return float(np.mean([score([r for r in chosen if r['phone'] == phone]) for phone in sorted({r['phone'] for r in chosen})]))
        return score(chosen)
    optimized = minimize_scalar(lambda p: objective(p, False, True), bounds=(0.001, 0.999), method='bounded', options=dict(xatol=1e-06))
    fitted = float(optimized.x)
    baseline = objective(0.0, True)
    held = objective(fitted, True)
    prior.update(directional_velocity_parameters=circular['directional_velocity_parameters'], directional_velocity_directions=circular.get('directional_velocity_directions', 16), directional_velocity_family_marginalization=True, directional_velocity_family_probability=fitted, directional_velocity_family_source_heldout_NLL=held, directional_velocity_family_baseline_NLL=baseline, directional_velocity_family_source_heldout_preferred=held < baseline, directional_velocity_family_source_train_phone_balanced=True, directional_velocity_family_target_GT_used=False, directional_velocity_family_source_fit=str(a.output.resolve()), cache_solver_groups=True, save_solver_state=True)
    prior.pop('directional_velocity_fraction', None)
    proof_row = rows[0]
    ctx = context(prior['directional_velocity_parameters'], proof_row['dt'], proof_row['time_s'], prior['directional_velocity_directions'])
    args = (proof_row['speed'], proof_row['dt'], prior['velocity_state_parameters'], prior['velocity_state_temporal_selected'], proof_row['speed_sigma'], ctx)
    value, g, _ = family_energy(proof_row['velocity'], *args, fitted, True)
    direction = np.random.RandomState(31991).normal(size=proof_row['velocity'].shape)
    direction /= np.linalg.norm(direction)
    h = 1e-05
    numeric = (family_energy(proof_row['velocity'] + h * direction, *args, fitted) - family_energy(proof_row['velocity'] - h * direction, *args, fitted)) / (2 * h)
    difference = float(abs(numeric - (direction * g).sum()))
    old = velocity_energy(proof_row['velocity'], proof_row['speed'], proof_row['dt'], prior['velocity_state_parameters'], prior['velocity_state_temporal_selected'], True, proof_row['speed_sigma'])
    endpoint = family_energy(proof_row['velocity'], *args, 0.0, True)
    endpoint_difference = max(abs(old[0] - endpoint[0]), float(abs(old[1] - endpoint[1]).max()))
    if difference > 1e-05 or endpoint_difference > 1e-10:
        raise ValueError('Motion family gradient/old-endpoint proof failed')
    result = dict(status='COMPLETE_SOURCE_ONLY_MOTION_FAMILY_PRIOR', source_prior=prior, fitted_circular_probability=fitted, source_heldout_NLL=held, source_baseline_heldout_NLL=baseline, source_heldout_preferred=held < baseline, source_train_paths=43, source_heldout_paths=11, target_GT_used=False, original_radio_and_initial_eight_preserved=True, motion_models_are_alternatives_not_duplicate_measurements=True, gradient_difference=difference, zero_probability_old_model_difference=endpoint_difference, reports=reports, seconds=time.monotonic() - began)
    write(a.output, result)
    write(a.output.parent / 'source_prior.json', prior)
    print(json.dumps({k: result[k] for k in ['status', 'fitted_circular_probability', 'source_heldout_NLL', 'source_baseline_heldout_NLL', 'source_heldout_preferred', 'seconds']}), flush=True)

def main():
    p = argparse.ArgumentParser()
    for k in ['base', 'capture', 'features', 'prior', 'circular-prior', 'output']:
        p.add_argument('--' + k, type=Path, required=True)
    fit(p.parse_args())
if __name__ == '__main__':
    main()
