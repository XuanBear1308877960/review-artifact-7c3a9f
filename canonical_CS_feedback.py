import json
import math
import numpy as np
from canonical_replay import ROOT
import trajectory_raw_hmm as raw_module
from trajectory_joint_RF import auxiliary_emission
ORIGINAL_ENERGY = raw_module.raw_hmm_energy

def joint_emission(distance, context):
    if 'auxiliary_family_probability' in context:
        raise ValueError('A4 family-mixture intervention needs explicit family responsibilities')
    p = context['parameters']
    raw = context['raw']
    sigma = context['sigma']
    error = raw[:, None] - p['gain'] * np.asarray(distance)[:, None] - p['anchor_mean'][context['anchor']] - p['alias'][None]
    error -= context.get('conditional_bias_m', np.zeros(len(raw)))[:, None]
    error = np.c_[error, raw - context['background']]
    density = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi) - np.log(sigma) - 2 * np.log1p(error * error / (3 * sigma * sigma))
    gradient = np.zeros_like(density)
    gradient[:, :3] = p['gain'] * 4 * error[:, :3] / (3 * sigma[:, :3] ** 2 + error[:, :3] ** 2)
    if 'auxiliary_parameters' in context:
        extra, negative_log_density_gradient = auxiliary_emission(distance, error, context, True)
        density += extra
        gradient -= negative_log_density_gradient
    return (density, gradient)

def fixed_energy(distance, context, derivative=False, bias_m=0.0):
    if 'canonical_A4_gamma' not in context:
        return ORIGINAL_ENERGY(distance, context, derivative, bias_m)
    if bias_m != 0.0:
        raise ValueError('A4 currently preserves the selected zero additional-bias model')
    density, g = joint_emission(distance, context)
    gamma = context['canonical_A4_gamma']
    value = context['canonical_A4_constant'] - float((gamma * density).sum())
    return (value, -(gamma * g).sum(-1), gamma) if derivative else value

def configure(solver):
    if not solver.prior.get('canonical_A4_no_WiFi_to_CS', False):
        return
    if np.any(np.asarray(solver.prior.get('raw_HMM_bias_sigma_m', [0.0] * 4))):
        raise ValueError('A4 must also freeze any nonzero extra range-bias posterior')
    cohort = solver.predictions[0].get('cohort')
    prefix = ROOT / 'prefix_capture'
    candidates = list(prefix.glob('*/' + solver.s['trajectory_id'] + '/predictions.json'))
    if len(candidates) != 1:
        raise ValueError('Unique prefix-only physical path required')
    rows = json.loads(candidates[0].read_text())
    times = np.r_[0.0, [(r['timestamp_ns'] - solver.s['start_ns']) / 1000000000.0 for r in rows]]
    modes = np.asarray([r['mode_xy_m'] for r in rows])
    weights = np.asarray(rows[-1]['mode_weights'])
    weights /= weights.sum()
    distances = []
    for k in range(modes.shape[1]):
        xy = np.vstack([solver.initial, modes[:, k]])
        cs_xy = np.stack([np.interp(solver.ct, times, xy[:, j]) for j in range(2)], -1)
        distances.append(np.linalg.norm(cs_xy - solver.raw_anchor, axis=-1))
    distances = np.asarray(distances)
    for ix, context in solver.hmm:
        probabilities = []
        constants = []
        for k in range(len(weights)):
            value, _, gamma = ORIGINAL_ENERGY(distances[k, ix], context, True)
            density, _ = joint_emission(distances[k, ix], context)
            probabilities.append(gamma)
            constants.append(value + float((gamma * density).sum()))
        context['canonical_A4_gamma'] = np.einsum('k,kts->ts', weights, np.asarray(probabilities))
        context['canonical_A4_constant'] = float(weights @ constants)
    solver.canonical_A4_distances = distances
    raw_module.raw_hmm_energy = fixed_energy

def diagnostics(solver):
    if not solver.prior.get('canonical_A4_no_WiFi_to_CS', False):
        return {}
    return dict(canonical_A4_fixed_CS_responsibilities=True, canonical_A4_original_joint_auxiliary_density_preserved=True, canonical_A4_range_values_modified=False, canonical_A4_CS_position_gradients_retained=True, canonical_A4_scope='Freeze the final raw/auxiliary CS HMM E-step to a common prefix-only CS/IMU beam; unchanged native proposal/context preprocessing is shared with Full')
