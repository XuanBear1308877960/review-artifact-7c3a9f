import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from canonical_replay import BASE, MaskedClockEngine, write
from trajectory_rts import RTSEngine
from trajectory_sequence import PathHead, feature_job
MODEL = BASE / 'runs/direction_v80/trajectory_shape_coverage_v4'
_MODEL = None
_INDEX = None
_LENGTHS = None
_FROZEN = None

class MaskedRTSEngine(RTSEngine, MaskedClockEngine):
    pass

def bundle():
    global _MODEL, _INDEX, _LENGTHS
    if _MODEL is None:
        saved = torch.load(MODEL / 'trained.pt', map_location='cpu', weights_only=False)
        _MODEL = PathHead(saved['sensor_dim'], saved.get('shape_only', False), saved.get('learn_shape_window', False), saved.get('shape_window_initial_s', 1.0), saved.get('global_offset', False)).eval()
        _MODEL.load_state_dict(saved['state_dict'])
        for p in _MODEL.parameters():
            p.requires_grad_(False)
        entries = json.loads((MODEL / 'feature_manifest.json').read_text())['entries']
        _INDEX = {(e['cohort'], e['segment']['trajectory_id']): i for i, e in enumerate(entries)}
        with np.load(MODEL / 'features.npz', allow_pickle=False) as z:
            _LENGTHS = z['lengths'].copy()
    return (_MODEL, _INDEX, _LENGTHS)

def rts_signature(engine):
    h = hashlib.sha256()
    for ns, mean, cov in engine.rts_states:
        h.update(np.asarray(ns, np.int64).tobytes())
        h.update(np.ascontiguousarray(mean).tobytes())
        h.update(np.ascontiguousarray(cov).tobytes())
    for entry in engine.rts_predictions:
        for a in entry:
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def finish(engine, folder, cohort, s, allow_frozen_reuse):
    global _FROZEN
    folder = Path(folder)
    capture = folder.parent.parent
    tid = s['trajectory_id']
    smoothed = engine.backward_means()
    weights = engine.log_weight.softmax(-1).cpu().numpy()
    rts = []
    for row in engine.outputs:
        x = smoothed[row['timestamp_ns']]
        rts.append(dict(row, xy_m=(weights[:, None] * x[:, :2]).sum(0).tolist(), mode_xy_m=x[:, :2].tolist(), mode_weights=weights.tolist(), inference_timing='OFFLINE_FULL_STATE_RTS', later_position_GT_used=False))
    proposal = capture / 'rts_mean'
    write(proposal / cohort / tid / 'predictions.json', rts)
    features = feature_job((str(BASE), str(capture), str(proposal), cohort, s, True))
    model, index, lengths = bundle()
    i = index[cohort, tid]
    n = len(features['times'])
    if n != int(lengths[i]):
        raise ValueError('The frozen shape prior has different observation membership')
    tensors = {}
    for key in ['radio', 'sensor', 'base', 'times']:
        a = np.zeros((len(lengths), int(lengths.max())) + features[key].shape[1:], np.float32)
        a[i, :n] = features[key]
        tensors[key] = torch.from_numpy(a)
    with torch.no_grad():
        prediction = model(tensors['radio'], tensors['sensor'], tensors['base'], tensors['times'], torch.from_numpy(lengths))[i, :n].numpy()
    reused = False
    forward_error = None
    feature_differences = {}
    if allow_frozen_reuse:
        if _FROZEN is None:
            with np.load(MODEL / 'features.npz', allow_pickle=False) as z:
                _FROZEN = {k: z[k].copy() for k in ['radio', 'sensor', 'base', 'times', 'ids']}
        lo = int(lengths[:i].sum())
        hi = lo + n
        identical = all((np.array_equal(features[k], _FROZEN[k][lo:hi]) for k in ['radio', 'sensor', 'base', 'times', 'ids']))
        for k in ['radio', 'sensor', 'base', 'times']:
            feature_differences[k] = float(abs(features[k] - _FROZEN[k][lo:hi]).max())
        if identical:
            old = json.loads((MODEL / 'predictions' / cohort / tid / 'predictions.json').read_text())
            if [r['observation_id'] for r in old] != features['ids'].tolist():
                raise ValueError('Frozen shape prior IDs changed')
            frozen = np.array([r['xy_m'] for r in old], np.float32)
            forward_error = float(abs(prediction - frozen).max())
            if forward_error > 2e-05:
                raise ValueError(('Frozen shape model no longer reproduces its cached source forward path', tid, forward_error))
            prediction = frozen
            reused = True
    rows = [dict(row, xy_m=xy.tolist(), inference_timing='OFFLINE_SOURCE_TRAINED_SEQUENCE', later_position_GT_used=False) for row, xy in zip(engine.outputs, prediction)]
    path = capture / 'source_motion_prior' / cohort / tid
    write(path / 'predictions.json', rows)
    report = dict(status='COMPLETED_FROZEN_RTS_SHAPE_GRU_PRIOR', source_model=str(MODEL), source_parameters=77187, target_GT_read=False, trained_weights_changed=False, context_query_values_read=False, full_input_feature_identity_memoized=reused, independent_forward_difference_m=forward_error, regenerated_feature_differences_from_frozen_inputs=feature_differences, full_data_cache_forbidden_in_omitted_query_contexts=not allow_frozen_reuse, sensor_features=features['sensor'].shape[-1], RTS_state_sha256=rts_signature(engine))
    write(path / 'completion.json', report)
    if os.environ.get('YOMO_CANONICAL_DIRECTION_SEED') == '1':
        report['frozen_direction_seed'] = direction_seed(engine, capture, cohort, s, features, prediction, allow_frozen_reuse)
        write(path / 'completion.json', report)
    return report
_DIRECTION = None
_DIRECTION_LENGTHS = None
_DIRECTION_FROZEN = None

def direction_seed(engine, capture, cohort, s, features, proposal, allow_frozen_reuse):
    global _DIRECTION, _DIRECTION_LENGTHS, _DIRECTION_FROZEN
    from scipy.linalg import expm
    from trajectory_direction_sequence import DirectionHead, viterbi
    root = BASE / 'runs/direction_v80/trajectory_direction_mixture_v1'
    if _DIRECTION is None:
        ck = torch.load(root / 'trained.pt', map_location='cpu', weights_only=False)
        model = DirectionHead(ck['sensor_dim']).eval()
        model.load_state_dict(ck['state_dict'])
        for p in model.parameters():
            p.requires_grad_(False)
        _DIRECTION = (model, ck['transition_generator'])
        with np.load(root / 'inputs.npz', allow_pickle=False) as z:
            _DIRECTION_LENGTHS = z['lengths'].copy()
    model, generator = _DIRECTION
    _, index, _ = bundle()
    i = index[cohort, s['trajectory_id']]
    t = features['times']
    grid = np.linspace(0, float(t[-1]), int(np.ceil(t[-1] / 0.2)) + 1, dtype=np.float32)

    def interpolate(values, initial=None):
        tt = np.r_[0, t] if initial is not None else t
        vv = np.vstack((initial, values)) if initial is not None else values
        return np.stack([np.interp(grid, tt, vv[:, j]) for j in range(vv.shape[1])], -1).astype(np.float32)
    radio = interpolate(features['radio'])
    sensor = interpolate(features['sensor'])
    initial = np.asarray(s['initial_position_xy_m'], np.float32)
    base = interpolate(np.asarray(proposal, np.float32), initial)
    old = interpolate(features['base'], initial)
    sensor[:, :2] = base / 10
    sensor[:, 4:6] = (base - 10 * sensor[:, 2:4]) / 2
    sensor[:, 8] = grid / 20
    sensor[:, 9] = np.diff(np.r_[0, grid]) / 0.5
    velocity = np.gradient(base, grid, axis=0)
    acceleration = np.gradient(velocity, grid, axis=0)
    sensor = np.concatenate((sensor, velocity / 3, acceleration / 5, (base - old) / 2), -1)
    n = len(grid)
    lengths = _DIRECTION_LENGTHS
    if n != int(lengths[i]):
        raise ValueError('Direction prior observation clock changed')
    row = dict(radio=radio, sensor=sensor, base=base, times=grid)
    tensors = {}
    for k, v in row.items():
        padded = np.zeros((len(lengths), int(lengths.max())) + v.shape[1:], np.float32)
        padded[i, :n] = v
        tensors[k] = torch.from_numpy(padded)
    with torch.no_grad():
        logits, candidates = model(tensors['radio'], tensors['sensor'], torch.from_numpy(lengths))
        transition = expm(generator * float(grid[1]))
        states = viterbi(logits[i:i + 1], transition[None], np.asarray([n]))
        velocity = torch.gather(candidates[i:i + 1], -2, states[..., None, None].expand(-1, -1, 1, 2)).squeeze(-2)
        result = model.fuse(velocity, tensors['base'][i:i + 1], tensors['times'][i:i + 1], torch.tensor([n]))[0, :n].numpy()
    times = np.asarray([(r['timestamp_ns'] - s['start_ns']) / 1000000000.0 for r in engine.outputs])
    xy = np.stack([np.interp(times, grid, result[:, j]) for j in range(2)], -1)
    reused = False
    forward_error = None
    differences = {}
    if allow_frozen_reuse:
        if _DIRECTION_FROZEN is None:
            with np.load(root / 'inputs.npz', allow_pickle=False) as z:
                _DIRECTION_FROZEN = {k: z[k].copy() for k in row}
        differences = {k: float(abs(v - _DIRECTION_FROZEN[k][i, :n]).max()) for k, v in row.items()}
        if all((np.array_equal(v, _DIRECTION_FROZEN[k][i, :n]) for k, v in row.items())):
            frozen = json.loads((root / 'predictions' / cohort / s['trajectory_id'] / 'predictions.json').read_text())
            if [r['observation_id'] for r in frozen] != features['ids'].tolist():
                raise ValueError('Direction prior IDs differ')
            old_xy = np.asarray([r['xy_m'] for r in frozen])
            forward_error = float(abs(xy - old_xy).max())
            if forward_error > 2e-05:
                raise ValueError(('Frozen directional model no longer reproduces', forward_error))
            xy = old_xy
            reused = True
    rows = [dict(r, xy_m=x.tolist(), inference_timing='OFFLINE_SOURCE_TRAINED_MAP_DIRECTIONS', later_position_GT_used=False) for r, x in zip(engine.outputs, xy)]
    out = Path(capture) / 'direction_motion_seed' / cohort / s['trajectory_id']
    write(out / 'predictions.json', rows)
    report = dict(status='COMPLETED_FROZEN_DIRECTION_GRU_VITERBI_SEED', source_checkpoint=str(root / 'trained.pt'), source_parameters=sum((p.numel() for p in model.parameters())), trained_weights_changed=False, target_GT_read=False, query_input_cache_forbidden=not allow_frozen_reuse, full_input_feature_identity_memoized=reused, independent_forward_difference_m=forward_error, feature_differences=differences)
    write(out / 'completion.json', report)
    return report
