import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from base_model import mlp
from base_evaluate import AVAILABLE_CPUS
from engine_utils import write
from trajectory_path import speed_evidence

class PathHead(nn.Module):

    def __init__(self, sensors, shape_only=False, learn_shape_window=False, shape_window_initial_s=1.0, global_offset=False):
        super().__init__()
        self.radio = mlp(220, 96, 32)
        self.sensor = mlp(sensors, 64, 48)
        self.sequence = nn.GRU(80, 48, batch_first=True, bidirectional=True)
        self.output = mlp(96, 48, 2, last_zero=True)
        self.shape_only = shape_only
        self.learn_shape_window = learn_shape_window
        self.global_offset = global_offset
        if global_offset:
            self.offset = mlp(96, 32, 2, last_zero=True)
        if learn_shape_window:
            self.shape_time_log = nn.Parameter(torch.tensor(math.log(shape_window_initial_s)))

    def forward(self, radio, sensor, base, times, lengths):
        features = torch.cat((self.radio(radio), self.sensor(sensor)), -1)
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        result, _ = self.sequence(packed)
        result, _ = pad_packed_sequence(result, batch_first=True, total_length=radio.shape[1])
        correction = 2.5 * self.output(result).tanh()
        correction = correction * (1 - (-times.clamp_min(0) / 0.5).exp())[..., None]
        if self.shape_only:
            valid = torch.arange(times.shape[1], device=times.device)[None] < lengths[:, None]
            window = self.shape_time_log.exp().clamp(0.5, 6.0) if self.learn_shape_window else 1.0
            weights = torch.cat((torch.full_like(times[:, :1], 0.1), times[:, 1:] - times[:, :-1]), 1).clamp(0.025, 0.5)
            kernel = (-0.5 * ((times[:, :, None] - times[:, None, :]) / window) ** 2).exp() * valid[:, None, :]
            if self.learn_shape_window:
                kernel = kernel * weights[:, None, :]
            kernel = kernel / kernel.sum(-1, keepdim=True).clamp_min(1e-06)
            correction = correction - torch.bmm(kernel, correction)
            elapsed = (times - times[:, :1]).clamp_min(0)
            correction = correction - correction[:, :1] * (-elapsed / 0.5).exp()[..., None]
        if self.global_offset:
            valid = torch.arange(times.shape[1], device=times.device)[None] < lengths[:, None]
            pooled = (result * valid[..., None]).sum(1) / lengths[:, None]
            offset = 1.5 * self.offset(pooled).tanh()
            correction = correction + offset[:, None] * (1 - (-times.clamp_min(0) / 1.0).exp())[..., None]
        return base + correction

def randomize_phone_coordinates(sensor):
    sensor = sensor.clone()
    token = sensor[..., 92:117].clone()
    q = torch.randn(len(sensor), 4, device=sensor.device)
    q = q / q.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-06)
    w, x, y, z = q.unbind(-1)
    B = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1).reshape(-1, 3, 3)
    columns = token[..., :6].reshape(*token.shape[:2], 3, 2)
    R = torch.cat((columns, torch.cross(columns[..., 0], columns[..., 1], dim=-1)[..., None]), -1)
    previous = torch.cat((R[:, :1], R[:, :-1]), 1)
    delta = previous.transpose(-1, -2) @ R
    rotated = B[:, None].transpose(-1, -2) @ R @ B[:, None]
    relative_delta = B[:, None].transpose(-1, -2) @ delta @ B[:, None]
    token[..., :6] = rotated[..., :2].reshape(*token.shape[:2], 6)
    angle = torch.atan2(relative_delta[..., 1, 0], relative_delta[..., 0, 0])
    token[..., 6] = angle.sin()
    token[..., 7] = angle.cos()
    for sl in [slice(8, 11), slice(11, 14), slice(17, 20)]:
        token[..., sl] = torch.einsum('bji,btj->bti', B, token[..., sl])
    token[..., 14:17] = torch.einsum('bji,btj->bti', B.square(), token[..., 14:17].square()).clamp_min(0).sqrt()
    sensor[..., 92:117] = token
    return sensor

def feature_job(task):
    base, capture, proposal, cohort, s, relative_imu = task
    base, capture, proposal = (Path(base), Path(capture), Path(proposal))
    torch.set_num_threads(1)
    folder = capture / cohort / s['trajectory_id']
    from support_label_barrier import deny_label_reads
    with deny_label_reads() as audit:
        with np.load(folder / 'trajectory_factors.npz', allow_pickle=False) as z:
            f = {k: z[k].copy() for k in z.files}
        with np.load(folder / 'CS_training_contexts.npz', allow_pickle=False) as z:
            cs = {k: z[k].copy() for k in z.files}
        with np.load(folder / 'PRE_forecasts.npz', allow_pickle=False) as z:
            pre = {k: z[k].copy() for k in ['rssi', 'mask', 'short_residual', 'weights', 'observation_ids']}
        native = json.loads((folder / 'predictions.json').read_text())
        proposed = json.loads((proposal / cohort / s['trajectory_id'] / 'predictions.json').read_text())
        times = f['times_ns']
        elapsed = (times - s['start_ns']) / 1000000000.0
        xy = np.asarray([p['xy_m'] for p in proposed], float)
        native_xy = np.asarray([p['xy_m'] for p in native], float)
        data = base / ('runs/ap_movement_v79_291/experiment/preprocessed' if cohort == 'AP_movement' else 'data/processed_rssi_causal_short_v1')
        speed, sigma, speed_valid, imu = speed_evidence(base, data, s, times, return_features=True)
        if AVAILABLE_CPUS:
            os.sched_setaffinity(0, AVAILABLE_CPUS)
        radio = logsumexp(f['radio_loglike'] + np.log(np.maximum(pre['weights'], 1e-20))[..., None], axis=1)
        radio = np.clip(radio - radio.max(-1, keepdims=True), -12, 0) / 6
        cs_features = []
        for ns in times:
            row = []
            for anchor in range(4):
                ix = np.where((cs['timestamp_ns'] <= ns) & (cs['anchor_index'] == anchor))[0]
                if not len(ix):
                    row.extend([0.0] * 10)
                    continue
                j = ix[-1]
                age = (ns - cs['timestamp_ns'][j]) / 1000000000.0
                dt = (cs['timestamp_ns'][j] - cs['timestamp_ns'][ix[-2]]) / 1000000000.0 if len(ix) > 1 else 0.0
                change = float(cs['range_m'][j] - cs['range_m'][ix[-2]]) if dt > 0.0001 else 0.0
                bias = float(cs['bias_mean'][j] @ cs['mode_weights'][j])
                row.extend([cs['range_m'][j] / 20, cs['features'][j, 0], bias / 5, min(age, 6) / 2, f['cs_sigma'][j] / 3, f['cs_pi'][j], change / 5, min(dt, 5) / 2, float(age <= 2.0), float(dt > 0.0001)])
            cs_features.append(row)
        sensors = np.concatenate((xy / 10, native_xy / 10, (xy - native_xy) / 2, np.broadcast_to(np.asarray(s['initial_position_xy_m']) / 10, (len(times), 2)), elapsed[:, None] / 20, np.diff(np.r_[0.0, elapsed])[:, None] / 0.5, pre['rssi'] / 100, pre['mask'].astype(float), (pre['short_residual'] * pre['weights'][..., None]).sum(1) / 10, np.asarray(cs_features), imu, np.stack((speed / 3, sigma / 2, speed_valid), -1)), axis=-1)
        if relative_imu:
            with np.load(folder / 'configured/rolling_pcrt_capture.npz', allow_pickle=False) as z:
                if not np.array_equal(z['observation_ids'], pre['observation_ids']):
                    raise ValueError('Relative IMU context IDs differ')
                token = z['motion_tokens'].copy()
            if token.shape[-1] != 41:
                raise ValueError('Unexpected causal motion token')
            sensors = np.concatenate((sensors, token[:, :25]), -1)
        assert np.isfinite(sensors).all() and (not audit['blocked_label_attempts'])
    return dict(cohort=cohort, trajectory_id=s['trajectory_id'], device=s['device'], radio=radio.astype(np.float32), sensor=sensors.astype(np.float32), base=xy.astype(np.float32), times=elapsed.astype(np.float32), ids=pre['observation_ids'])

def features(a):
    began = time.monotonic()
    cohorts = json.loads((a.capture / 'segments.json').read_text())
    entries = [(c, s) for c, segments in cohorts.items() for s in segments]
    jobs = [(str(a.base), str(a.capture), str(a.proposal), c, s, a.relative_imu) for c, s in entries]
    with ProcessPoolExecutor(max_workers=24) as pool:
        rows = list(pool.map(feature_job, jobs))
    lengths = np.asarray([len(r['times']) for r in rows])
    a.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output / 'features.npz', lengths=lengths, **{k: np.concatenate([r[k] for r in rows]) for k in ['radio', 'sensor', 'base', 'times', 'ids']})
    write(a.output / 'feature_manifest.json', dict(status='COMPLETE_LABEL_GUARDED_SEQUENCE_FEATURES', entries=[dict(cohort=c, segment=s) for c, s in entries], input_labels_opened=False, device_ID_is_input=False, absolute_phone_or_map_heading_is_input=False, offline_full_segment=True, relative_IMU_context_included=a.relative_imu, absolute_phone_heading_used_as_map_bearing=False, seconds=time.monotonic() - began))
    print(json.dumps(dict(stage='features', trajectories=len(rows), seconds=time.monotonic() - began)), flush=True)

def load_arrays(folder):
    with np.load(folder / 'features.npz', allow_pickle=False) as z:
        flat = {k: z[k].copy() for k in z.files}
    lengths = flat.pop('lengths')
    maximum = int(lengths.max())
    arrays = {}
    boundaries = np.r_[0, np.cumsum(lengths)]
    for k in ['radio', 'sensor', 'base', 'times']:
        data = flat[k]
        padded = np.zeros((len(lengths), maximum) + data.shape[1:], np.float32)
        for i, n in enumerate(lengths):
            padded[i, :n] = data[boundaries[i]:boundaries[i + 1]]
        arrays[k] = padded
    arrays['lengths'] = lengths
    return (arrays, flat, boundaries)

def train(a):
    torch.set_num_threads(1)
    torch.manual_seed(81102 if a.shape_loss else 81101)
    np.random.seed(81102 if a.shape_loss else 81101)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    arrays, flat, bounds = load_arrays(a.output)
    manifest = json.loads((a.output / 'feature_manifest.json').read_text())
    entries = manifest['entries']
    source = [i for i, e in enumerate(entries) if e['cohort'] == 'source_train']
    if len(source) != 54:
        raise ValueError('Unexpected source cohort')
    labels = {}
    target = np.zeros_like(arrays['base'])
    valid = np.zeros(arrays['times'].shape, bool)
    pairs = []
    for i in source:
        s = entries[i]['segment']
        name = s['device']
        if name not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target in training labels')
        if name not in labels:
            labels[name] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / name / 'evaluation_labels.json').read_text())
        for j, oid in enumerate(flat['ids'][bounds[i]:bounds[i + 1]]):
            if str(oid) in labels[name]:
                target[i, j] = labels[name][str(oid)]['gt_xy_m']
                valid[i, j] = j >= 4
    pair_index = []
    pair_valid = []
    for horizon in [0.5, 1.0, 2.0]:
        ix = np.zeros(arrays['times'].shape, np.int64)
        pv = np.zeros_like(valid)
        for i in source:
            n = arrays['lengths'][i]
            t = arrays['times'][i, :n]
            j = np.searchsorted(t, t + horizon).clip(0, n - 1)
            ix[i, :n] = j
            pv[i, :n] = valid[i, :n] & valid[i, j] & (t[j] - t >= horizon * 0.8)
        pair_index.append(ix)
        pair_valid.append(pv)
    device = torch.device('cuda:0')
    tensors = {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}
    y = torch.from_numpy(target).to(device)
    mask = torch.from_numpy(valid).to(device)
    indexes = [torch.from_numpy(v).to(device) for v in pair_index]
    pmasks = [torch.from_numpy(v).to(device) for v in pair_valid]
    holdout = [i for ordinal, i in enumerate(source) if ordinal % 5 == 0]
    training = [i for i in source if i not in holdout]
    initial_window = float(np.median([entries[i]['segment']['duration_s'] for i in training])) / 4 if a.learn_shape_window else 1.0
    shape_readout = a.shape_loss and (not a.full_readout)
    model = PathHead(arrays['sensor'].shape[-1], shape_readout, a.learn_shape_window, initial_window, a.global_offset).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.003)
    train_indices = torch.tensor(training, device=device)
    val_indices = torch.tensor(holdout, device=device)

    def forward(index, augment=False):
        radio = tensors['radio'][index]
        sensor = tensors['sensor'][index]
        base = tensors['base'][index]
        if augment:
            temperature = torch.empty(len(index), 1, 1, device=device).uniform_(0.8, 1.25)
            radio = radio * temperature
            if not manifest.get('source_prediction_geometry_only', False):
                perturbation = torch.randn(len(index), 1, 2, device=device) * 0.2
                perturbation = perturbation * (1 - (-tensors['times'][index] / 0.7).exp())[..., None]
                base = base + perturbation
                sensor = sensor.clone()
                sensor[..., :2] = base / 10
                sensor[..., 4:6] = (base - 10 * sensor[..., 2:4]) / 2
            if a.motion_frame_randomization:
                sensor = randomize_phone_coordinates(sensor)
        return model(radio, sensor, base, tensors['times'][index], tensors['lengths'][index])

    def losses(pred, index):
        v = mask[index]
        truth = y[index]
        if a.robust_location_loss:
            e2 = (pred[v] - truth[v]).square().sum(-1)
            loss = 1.5 * torch.log1p(e2 / 0.75 ** 2).mean() + 0.1 * F.smooth_l1_loss(pred[v], truth[v], beta=0.75)
        else:
            loss = F.smooth_l1_loss(pred[v], truth[v], beta=0.75)
        displacement = []
        for ix, pv in zip(indexes, pmasks):
            gather = ix[index][..., None].expand(-1, -1, 2)
            dp = torch.gather(pred, 1, gather) - pred
            dg = torch.gather(truth, 1, gather) - truth
            m = pv[index]
            shape = 1.5 * torch.log1p((dp[m] - dg[m]).square().sum(-1) / 0.4 ** 2).mean() if a.robust_location_loss else F.smooth_l1_loss(dp[m], dg[m], beta=0.4)
            pl = dp.square().sum(-1).clamp_min(1e-06).sqrt()
            gl = dg.square().sum(-1).clamp_min(1e-06).sqrt()
            length = F.smooth_l1_loss(pl[m], gl[m], beta=0.3)
            loss = loss + 0.5 * shape + 0.2 * length
            displacement.append(torch.linalg.vector_norm(dp[m] - dg[m], dim=-1).mean())
            if a.shape_loss:
                moving = m & (gl >= 0.5)
                cosine = (dp * dg).sum(-1) / (pl * gl).clamp_min(1e-05)
                loss = loss + 0.1 * (1 - cosine[moving].clamp(-1, 1)).mean()
        if a.shape_loss:
            pairmask = v[:, :, None] & v[:, None, :] & ((tensors['times'][index, :, None] - tensors['times'][index, None, :]).abs() <= 1.0)
            distances = (pred[:, :, None, :] - truth[:, None, :, :]).square().sum(-1).masked_fill(~pairmask, 10000.0)
            nearest_pred = distances.min(2).values.clamp_min(1e-06).sqrt()
            nearest_gt = distances.min(1).values.clamp_min(1e-06).sqrt()
            coverage = 0.5 * (nearest_pred[v].mean() + nearest_gt[v].mean())
            if a.robust_location_loss:
                robust_coverage = 0.5 * (torch.log1p(nearest_pred[v].square() / 0.75 ** 2).mean() + torch.log1p(nearest_gt[v].square() / 0.75 ** 2).mean())
                loss = loss + 0.6 * robust_coverage + 0.25 * torch.log1p(torch.quantile(nearest_gt[v], 0.9).square() / 0.75 ** 2)
            else:
                loss = loss + 0.6 * coverage + 0.25 * torch.quantile(nearest_gt[v], 0.9)
        loss = loss + 0.015 * (pred - tensors['base'][index]).square().mean()
        return (loss, torch.stack(displacement).mean())

    def validate():
        model.eval()
        with torch.no_grad():
            pred = forward(val_indices)
            _, disp = losses(pred, val_indices)
            error = torch.linalg.vector_norm(pred - y[val_indices], dim=-1)[mask[val_indices]]
            row = dict(median_m=float(error.median()), mean_m=float(error.mean()), p90_m=float(torch.quantile(error, 0.9)), p95_m=float(torch.quantile(error, 0.95)), displacement_error_m=float(disp))
            row['selection_score'] = row['mean_m'] + 0.25 * row['p95_m'] + 0.4 * row['displacement_error_m']
            if a.robust_location_loss:
                row['selection_score'] = row['median_m'] + 0.35 * row['mean_m'] + 0.15 * row['p90_m'] + 0.15 * row['p95_m'] + 0.3 * row['displacement_error_m']
            if a.shape_loss:
                truth = y[val_indices]
                v = mask[val_indices]
                pairmask = v[:, :, None] & v[:, None, :] & ((tensors['times'][val_indices, :, None] - tensors['times'][val_indices, None, :]).abs() <= 1.0)
                distance = (pred[:, :, None, :] - truth[:, None, :, :]).square().sum(-1).masked_fill(~pairmask, 10000.0)
                directed = distance.min(1).values.clamp_min(1e-06).sqrt()[v]
                row['curve_coverage_mean_m'] = float(directed.mean())
                row['curve_coverage_p90_m'] = float(torch.quantile(directed, 0.9))
                row['selection_score'] += 0.35 * row['curve_coverage_mean_m'] + 0.15 * row['curve_coverage_p90_m'] if a.robust_location_loss else 0.6 * row['curve_coverage_mean_m'] + 0.25 * row['curve_coverage_p90_m']
        return row
    began = time.monotonic()
    best = float('inf')
    trace = []
    for update in range(601):
        if update % a.validation_interval == 0 or update == 600:
            row = validate()
            row.update(update=update, seconds=time.monotonic() - began)
            trace.append(row)
            if row['selection_score'] < best:
                best = row['selection_score']
                torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, sensor_dim=arrays['sensor'].shape[-1], source_training_devices=sorted(labels), shape_only=shape_readout, global_offset=a.global_offset, motion_frame_randomization=a.motion_frame_randomization, robust_location_loss=a.robust_location_loss, learn_shape_window=a.learn_shape_window, shape_window_initial_s=initial_window, target_labels_used=False, source_selection=row, offline=True), a.output / 'trained.pt')
            print(json.dumps(row), flush=True)
        if update == 600:
            break
        model.train()
        index = train_indices[torch.randint(len(train_indices), (128,), device=device)]
        optimizer.zero_grad(set_to_none=True)
        pred = forward(index, True)
        loss, _ = losses(pred, index)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite source trajectory loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
    write(a.output / 'training_complete.json', dict(status='COMPLETE_SOURCE_ONLY_FULL_TRAJECTORY_TRAINING', train_source_trajectories=len(training), heldout_source_trajectories=len(holdout), optimizer_updates=600, batch_size=128, new_parameters=sum((p.numel() for p in model.parameters())), source_devices=sorted(labels), target_labels_opened=False, trace=trace, curve_coverage_loss=a.shape_loss, shape_only_readout=shape_readout, global_offset=a.global_offset, motion_frame_randomization=a.motion_frame_randomization, robust_location_loss=a.robust_location_loss, learned_shape_window_s=float(model.shape_time_log.exp().clamp(0.5, 6.0)) if a.learn_shape_window else None, future_sensors_used=True, phone_heading_used=False, seconds=time.monotonic() - began, GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated()))

def predict(a):
    torch.set_num_threads(1)
    arrays, flat, bounds = load_arrays(a.output)
    manifest = json.loads((a.output / 'feature_manifest.json').read_text())
    saved = torch.load(a.output / 'trained.pt', map_location='cpu', weights_only=False)
    model = PathHead(saved['sensor_dim'], saved.get('shape_only', False), saved.get('learn_shape_window', False), saved.get('shape_window_initial_s', 1.0), saved.get('global_offset', False)).eval()
    model.load_state_dict(saved['state_dict'])
    from support_label_barrier import deny_label_reads
    root = a.output / 'predictions'
    with deny_label_reads() as audit, torch.no_grad():
        tensors = {k: torch.from_numpy(v) for k, v in arrays.items()}
        result = model(tensors['radio'], tensors['sensor'], tensors['base'], tensors['times'], tensors['lengths']).numpy()
        for i, e in enumerate(manifest['entries']):
            c, s = (e['cohort'], e['segment'])
            source = a.capture / c / s['trajectory_id']
            folder = root / c / s['trajectory_id']
            folder.mkdir(parents=True, exist_ok=True)
            original = json.loads((source / 'predictions.json').read_text())
            assert len(original) == arrays['lengths'][i]
            predictions = []
            for p, xy in zip(original, result[i, :len(original)]):
                row = dict(p, xy_m=xy.tolist(), inference_timing='OFFLINE_SOURCE_TRAINED_SEQUENCE', later_position_GT_used=False)
                predictions.append(row)
            write(folder / 'predictions.json', predictions)
            for name in ['PRE_forecasts.npz', 'CS_diagnostics.json', 'WiFi_quality_diagnostics.json']:
                shutil.copyfile(str(source / name), str(folder / name))
            if 'selected_calibration_mode' in e:
                with np.load(folder / 'PRE_forecasts.npz', allow_pickle=False) as z:
                    forecast = {k: z[k].copy() for k in z.files}
                mode = int(e['selected_calibration_mode'])
                forecast['weights'][:] = 0.0
                forecast['weights'][:, mode] = 1.0
                np.savez_compressed(folder / 'PRE_forecasts.npz', **forecast)
        assert not audit['blocked_label_attempts']
    write(root / 'prediction_complete.json', dict(status='COMPLETE_SENSOR_ONLY_SEQUENCE_INFERENCE', trajectories=len(manifest['entries']), later_GT=False, offline_future_sensors=True, all_original_IDs_retained=True, offline_selected_calibration_branch=any(('selected_calibration_mode' in e for e in manifest['entries']))))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['features', 'train', 'predict'], required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--proposal', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lock', type=Path)
    p.add_argument('--shape-loss', action='store_true')
    p.add_argument('--learn-shape-window', action='store_true')
    p.add_argument('--relative-imu', action='store_true')
    p.add_argument('--global-offset', action='store_true')
    p.add_argument('--motion-frame-randomization', action='store_true')
    p.add_argument('--full-readout', action='store_true')
    p.add_argument('--robust-location-loss', action='store_true')
    p.add_argument('--validation-interval', type=int, default=50)
    a = p.parse_args()
    if a.stage == 'train':
        if not a.lock:
            raise ValueError('GPU execution lock required')
        with a.lock.open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(a)
    elif a.stage == 'features':
        features(a)
    else:
        predict(a)
if __name__ == '__main__':
    main()
