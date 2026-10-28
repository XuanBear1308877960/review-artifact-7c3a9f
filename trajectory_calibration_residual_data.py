import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import pickle
import time
import numpy as np
from scipy.linalg import solve_triangular
import torch
from base_evaluate import AVAILABLE_CPUS
from engine_utils import write
from trajectory_axial_motion import feature_speeds
from trajectory_temporal_radio import TemporalJointPath

def descriptors(f, points, context, ap):
    _, features = f.basis_inputs_ap(points, context, ap)
    delta = points[:, None] - f.xy[None]
    d2 = np.square(delta).sum(-1) + f.radius2
    w = f.mask[:, ap].T / d2
    den = w.sum(-1)
    values = f.values[:, ap, :].transpose(1, 0, 2)
    component = (w[..., None] * values).sum(1) / den[:, None]
    dw = -2 * delta / d2[..., None] * w[..., None]
    dc = np.einsum('mgz,mgc->mcz', dw, values - component[:, None]) / den[:, None, None]
    offset = points - f.centres[ap]
    radius = np.square(offset).sum(-1) + 1.0
    ds = np.stack((offset / (math.log(10) * radius[:, None]), dc[:, 0] / 50, dc[:, 1] / 10, dc[:, 2] / (1 + component[:, 2, None]) / 5, dc[:, 3] * ((component[:, 3] > 0.01) & (component[:, 3] < 0.99))[:, None]), -2)
    directions = np.concatenate((np.broadcast_to(np.eye(2) / 10, (len(points), 2, 2)), ds, np.zeros((len(points), 42, 2))), -2)
    return (features, directions)

def job(task):
    base, capture, parent, learned, output, prior, s, evidence, report = task
    base, capture, parent, learned, output = map(Path, [base, capture, parent, learned, output])
    torch.set_num_threads(1)
    if AVAILABLE_CPUS:
        os.sched_setaffinity(0, AVAILABLE_CPUS)
    if s['device'] not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
        raise ValueError('Target in source basis preparation')
    from support_label_barrier import deny_label_reads
    with deny_label_reads() as audit:
        states = []
        for group in [0, 2, 4, 6]:
            with (parent / '.solver_groups/source_train' / s['trajectory_id'] / ('group_%d.pkl' % group)).open('rb') as handle:
                states.append(pickle.load(handle)['result'])
        best = min(states, key=lambda r: r['diagnostic']['objective'])
        x = best['optimization_state']['vector']
        folder = capture / 'source_train' / s['trajectory_id']
        rows = json.loads((learned / 'source_train' / s['trajectory_id'] / 'predictions.json').read_text())
        xy = np.asarray([r['xy_m'] for r in rows])
        speed = (evidence['speed'], evidence['sigma'], np.ones(len(xy), bool))
        solver = TemporalJointPath(folder, s, prior, speed, False, xy, force_context=evidence)
        if abs(solver.axial_objective(x, 0)[0] - best['diagnostic']['objective']) > 1e-08 * (1 + abs(best['diagnostic']['objective'])):
            raise ValueError('Basis feature model differs from the native parent')
        q, _ = solver.unpack(x[:-1])
        model = solver.observations
        points = solver.raw_spline @ q
        field = solver.radio_field
        while type(field).__name__ != 'FiniteRadioField':
            if not hasattr(field, 'base'):
                raise ValueError('Keep the frozen finite device descriptor')
            field = field.base
        features, directions = descriptors(field, points, model.raw_context(), model.ap)
        direction = np.array([0.6, 0.8])
        h = 1e-05
        numeric = (field.basis_inputs_ap(points + h * direction, model.raw_context(), model.ap)[1] - field.basis_inputs_ap(points - h * direction, model.raw_context(), model.ap)[1]) / (2 * h)
        difference = float(abs(numeric - directions @ direction).max())
        if difference > 5e-05:
            raise ValueError('Device descriptor position derivative differs')
        W = np.zeros((sum((len(ix) for _, ix, _ in solver.radio_blocks)), len(points)))
        offset = 0
        for ap, ix, L in solver.radio_blocks:
            W[offset:offset + len(ix), model.bounds[ap]:model.bounds[ap + 1]] = solve_triangular(L, model.S[ap][ix], lower=True, check_finite=False)
            offset += len(ix)
        target = output / 'source_features' / (s['trajectory_id'] + '.npz')
        target.parent.mkdir(parents=True, exist_ok=True)
        scale = np.ones(4) if field.strength is None else 1 - 0.85 * abs(field.strength)
        np.savez_compressed(target, raw_features=features, raw_feature_derivative=directions, whitening_operator=W, raw_position_spline=solver.raw_spline[:, 1:], output_scale=scale, observation_ids=model.pre['observation_ids'], source_sensitivity_path=np.asarray(report['path']))
        assert not audit['blocked_label_attempts']
    return dict(report, calibration_features_path=str(target.resolve()), descriptor_gradient_difference=difference, reference_positions_read=False, raw_samples=len(points), descriptor_dimensions=features.shape[-1])

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'features', 'parent', 'learned-path', 'data-root', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=96)
    a = p.parse_args()
    began = time.monotonic()
    meta = json.loads((a.data_root / 'data_complete.json').read_text())
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    if [s['trajectory_id'] for s in segments] != [r['trajectory_id'] for r in meta['reports']]:
        raise ValueError('Source feature order changed')
    evidence = feature_speeds(a.features)
    tasks = [(str(a.base), str(a.capture), str(a.parent), str(a.learned_path), str(a.output), meta['source_prior'], s, evidence['source_train', s['trajectory_id']], r) for s, r in zip(segments, meta['reports'])]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        reports = list(pool.map(job, tasks))
    write(a.output / 'data_complete.json', dict(status='COMPLETE_SOURCE_CALIBRATION_RESIDUAL_FEATURES_NOT_TRAINING', reports=reports, source_prior=meta['source_prior'], source_sensitivity_root=str(a.data_root.resolve()), source_train_trajectories=43, source_heldout_trajectories=11, target_data_used=False, reference_positions_read_for_descriptors=False, intended_model='49 to32 to4 small zero-initialized residual MLP on nonlinear device basis only; M1 mean frozen', seconds=time.monotonic() - began))
    print(json.dumps(dict(source_trajectories=len(reports), descriptor_dimensions=reports[0]['descriptor_dimensions'], maximum_descriptor_gradient_difference=max((r['descriptor_gradient_difference'] for r in reports)), seconds=time.monotonic() - began)), flush=True)
if __name__ == '__main__':
    main()
