import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.linalg import expm
from scipy.ndimage import gaussian_filter1d
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from base_model import mlp
from engine_utils import write
from trajectory_sequence import load_arrays

class DirectionHead(nn.Module):

    def __init__(self, sensors):
        super().__init__()
        self.radio = mlp(220, 96, 32)
        self.sensor = mlp(sensors, 64, 48)
        self.sequence = nn.GRU(80, 48, batch_first=True, bidirectional=True)
        self.direction = mlp(96, 48, 9)
        self.speed = mlp(96, 48, 8)
        self.angle = mlp(96, 48, 8, last_zero=True)
        self.shape_time_log = nn.Parameter(torch.tensor(math.log(3.0)))

    def forward(self, radio, sensor, lengths):
        x = torch.cat((self.radio(radio), self.sensor(sensor)), -1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.sequence(packed)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=radio.shape[1])
        logits = self.direction(x)
        speed = 3 * torch.sigmoid(self.speed(x))
        angle = torch.arange(8, device=x.device) * math.pi / 4 + math.pi / 8 * self.angle(x).tanh()
        velocity = speed[..., None] * torch.stack((angle.cos(), angle.sin()), -1)
        velocity = torch.cat((velocity, torch.zeros_like(velocity[..., :1, :])), -2)
        return (logits, velocity)

    def fuse(self, velocity, base, times, lengths):
        dt = torch.cat((times[:, :1] * 0, times[:, 1:] - times[:, :-1]), 1)
        integrated = torch.cumsum(0.5 * (velocity + torch.cat((velocity[:, :1], velocity[:, :-1]), 1)) * dt[..., None], 1)
        difference = integrated - (base - base[:, :1])
        valid = torch.arange(times.shape[1], device=times.device)[None] < lengths[:, None]
        sigma = self.shape_time_log.exp().clamp(0.5, 8.0)
        kernel = (-0.5 * ((times[:, :, None] - times[:, None, :]) / sigma) ** 2).exp() * valid[:, None, :]
        kernel = kernel / kernel.sum(-1, keepdim=True).clamp_min(1e-06)
        correction = difference - torch.bmm(kernel, difference)
        correction = correction - correction[:, :1] * (-times / 0.5).exp()[..., None]
        return base + correction

def prepare(a):
    source_arrays, flat, bounds = load_arrays(a.features)
    manifest = json.loads((a.features / 'feature_manifest.json').read_text())
    entries = manifest['entries']
    rows = []
    from support_label_barrier import deny_label_reads
    with deny_label_reads() as audit:
        for i, e in enumerate(entries):
            s = e['segment']
            n = int(source_arrays['lengths'][i])
            t = source_arrays['times'][i, :n]
            pred = json.loads((a.proposal / e['cohort'] / s['trajectory_id'] / 'predictions.json').read_text())
            if [p['observation_id'] for p in pred] != flat['ids'][bounds[i]:bounds[i + 1]].tolist():
                raise ValueError('Proposal IDs differ from feature IDs')
            p = np.asarray([r['xy_m'] for r in pred], np.float32)
            original = source_arrays['base'][i, :n].copy()
            grid = np.linspace(0, float(t[-1]), int(np.ceil(t[-1] / 0.2)) + 1, dtype=np.float32)

            def interpolate(values, initial=None):
                tt = np.r_[0, t] if initial is not None else t
                vv = np.vstack((initial, values)) if initial is not None else values
                return np.stack([np.interp(grid, tt, vv[:, j]) for j in range(vv.shape[1])], -1).astype(np.float32)
            radio = interpolate(source_arrays['radio'][i, :n])
            sensor = interpolate(source_arrays['sensor'][i, :n])
            base = interpolate(p, np.asarray(s['initial_position_xy_m'], np.float32))
            old = interpolate(original, np.asarray(s['initial_position_xy_m'], np.float32))
            sensor[:, :2] = base / 10
            sensor[:, 4:6] = (base - 10 * sensor[:, 2:4]) / 2
            sensor[:, 8] = grid / 20
            sensor[:, 9] = np.diff(np.r_[0, grid]) / 0.5
            velocity = np.gradient(base, grid, axis=0)
            acceleration = np.gradient(velocity, grid, axis=0)
            sensor = np.concatenate((sensor, velocity / 3, acceleration / 5, (base - old) / 2), -1)
            rows.append(dict(radio=radio, sensor=sensor, base=base, times=grid))
        assert not audit['blocked_label_attempts']
    maximum = max((len(r['times']) for r in rows))
    arrays = {}
    for k in ['radio', 'sensor', 'base', 'times']:
        arrays[k] = np.zeros((len(rows), maximum) + rows[0][k].shape[1:], np.float32)
        for i, r in enumerate(rows):
            arrays[k][i, :len(r['times'])] = r[k]
    arrays['lengths'] = np.asarray([len(r['times']) for r in rows], np.int64)
    a.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output / 'inputs.npz', **arrays)
    write(a.output / 'feature_manifest.json', dict(entries=entries, source_feature_directory=str(a.features), proposal_directory=str(a.proposal), input_labels_opened=False, uniform_time_interval_at_most_s=0.2, absolute_phone_heading_used_as_map_bearing=False, offline_future_sensor_context=True))

def viterbi(logits, transition, lengths):
    logp = torch.log_softmax(logits, -1).detach().cpu().numpy()
    result = np.zeros(logp.shape[:2], np.int64)
    log_transition = np.log(np.maximum(transition, 1e-20))
    for i, n in enumerate(lengths):
        table = logp[i, 0] - math.log(9)
        back = []
        for j in range(1, int(n)):
            candidate = table[:, None] + log_transition[i]
            ix = candidate.argmax(0)
            back.append(ix)
            table = candidate[ix, np.arange(9)] + logp[i, j]
        state = int(table.argmax())
        result[i, int(n) - 1] = state
        for j in range(int(n) - 2, -1, -1):
            state = int(back[j][state])
            result[i, j] = state
    return torch.from_numpy(result).to(logits.device)

def source_targets(a, arrays, entries):
    source = [i for i, e in enumerate(entries) if e['cohort'] == 'source_train']
    holdout = [i for ordinal, i in enumerate(source) if ordinal % 5 == 0]
    training = [i for i in source if i not in holdout]
    truth = np.zeros_like(arrays['base'])
    valid = np.zeros_like(arrays['times'], bool)
    labels = {}
    for i in source:
        s = entries[i]['segment']
        name = s['device']
        if name not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target in training labels')
        if name not in labels:
            labels[name] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / name / 'evaluation_labels.json').read_text())
        pred = json.loads((a.capture / 'source_train' / s['trajectory_id'] / 'predictions.json').read_text())
        rows = [r for r in pred if r['observation_id'] in labels[name]]
        t = np.r_[0, [(r['timestamp_ns'] - s['start_ns']) / 1000000000.0 for r in rows]]
        xy = np.vstack((s['initial_position_xy_m'], [labels[name][r['observation_id']]['gt_xy_m'] for r in rows]))
        n = arrays['lengths'][i]
        grid = arrays['times'][i, :n]
        truth[i, :n] = np.stack([np.interp(grid, t, xy[:, j]) for j in range(2)], -1)
        threshold = (pred[min(4, len(pred) - 1)]['timestamp_ns'] - s['start_ns']) / 1000000000.0
        valid[i, :n] = (grid >= threshold) & (grid <= t[-1])
    velocity = np.zeros_like(truth)
    for i in source:
        n = arrays['lengths'][i]
        grid = arrays['times'][i, :n]
        smooth = gaussian_filter1d(truth[i, :n], 0.25 / float(grid[1]), axis=0, mode='nearest')
        velocity[i, :n] = np.gradient(smooth, grid, axis=0)
    speed = np.linalg.norm(velocity, axis=-1)
    angle = np.arctan2(velocity[..., 1], velocity[..., 0])
    classes = np.floor((angle + math.pi / 8) / (math.pi / 4)).astype(int) % 8
    classes[speed < 0.2] = 8
    exposure = np.zeros(2)
    rotation = np.zeros(8)
    stop = 0.0
    start = 0.0
    for i in training:
        n = arrays['lengths'][i]
        dt = float(arrays['times'][i, 1])
        for j in range(n - 1):
            if not (valid[i, j] and valid[i, j + 1]):
                continue
            c, d = (classes[i, j], classes[i, j + 1])
            exposure[int(c == 8)] += dt
            if c != d:
                if c == 8:
                    start += 1
                elif d == 8:
                    stop += 1
                else:
                    rotation[(d - c) % 8] += 1
    q = np.zeros((9, 9))
    for c in range(8):
        for delta in range(1, 8):
            q[c, (c + delta) % 8] = (rotation[delta] + 0.1) / max(exposure[0], 1.0)
        q[c, 8] = (stop + 0.1) / max(exposure[0], 1.0)
    q[8, :8] = (start + 0.1) / max(exposure[1], 1.0) / 8
    q[np.arange(9), np.arange(9)] = -q.sum(-1)
    return (source, training, holdout, truth, valid, velocity, classes, q)

def train(a):
    torch.set_num_threads(1)
    torch.manual_seed(81205)
    np.random.seed(81205)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    with np.load(a.output / 'inputs.npz', allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    entries = json.loads((a.output / 'feature_manifest.json').read_text())['entries']
    source, training, holdout, truth, valid, velocity, classes, q = source_targets(a, arrays, entries)
    transition = np.stack([expm(q * float(arrays['times'][i, 1])) for i in range(len(entries))])
    device = torch.device('cuda:0')
    x = {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}
    y = torch.from_numpy(truth).to(device)
    v = torch.from_numpy(velocity).to(device)
    mask = torch.from_numpy(valid).to(device)
    state = torch.from_numpy(classes).to(device)
    model = DirectionHead(arrays['sensor'].shape[-1]).to(device)
    saved = torch.load(a.features / 'trained.pt', map_location='cpu', weights_only=False)['state_dict']
    own = model.state_dict()
    for k in own:
        if k in saved and own[k].shape == saved[k].shape:
            own[k] = saved[k]
    own['sensor.0.weight'][:, :saved['sensor.0.weight'].shape[1]] = saved['sensor.0.weight']
    own['sensor.0.weight'][:, saved['sensor.0.weight'].shape[1]:] = 0
    model.load_state_dict(own)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.003)
    train_index = torch.tensor(training, device=device)
    val_index = torch.tensor(holdout, device=device)

    def forward(index, hard=False):
        logits, candidates = model(x['radio'][index], x['sensor'][index], x['lengths'][index])
        if hard:
            states = viterbi(logits, transition[index.cpu().numpy()], arrays['lengths'][index.cpu().numpy()])
            chosen = torch.gather(candidates, -2, states[..., None, None].expand(-1, -1, 1, 2)).squeeze(-2)
        else:
            chosen = (logits.softmax(-1)[..., None] * candidates).sum(-2)
        pred = model.fuse(chosen, x['base'][index], x['times'][index], x['lengths'][index])
        return (pred, logits, candidates, chosen)

    def losses(pred, logits, candidates, chosen, index):
        m = mask[index]
        target = y[index]
        truev = v[index]
        c = state[index]
        classification = F.cross_entropy(logits[m], c[m])
        correct = torch.gather(candidates, -2, c[..., None, None].expand(-1, -1, 1, 2)).squeeze(-2)
        moving = m & (c != 8)
        velocity_loss = F.smooth_l1_loss(correct[moving], truev[moving], beta=0.3)
        loss = classification + 0.8 * velocity_loss + 0.5 * F.smooth_l1_loss(chosen[m], truev[m], beta=0.3)
        loss += F.smooth_l1_loss(pred[m], target[m], beta=0.75)
        for hop in [3, 5, 10]:
            eligible = m[:, hop:] & m[:, :-hop]
            dp = pred[:, hop:] - pred[:, :-hop]
            dg = target[:, hop:] - target[:, :-hop]
            loss += 0.4 * F.smooth_l1_loss(dp[eligible], dg[eligible], beta=0.4)
        band = (x['times'][index, :, None] - x['times'][index, None, :]).abs() <= 1.0
        d = (pred[:, :, None] - target[:, None]).square().sum(-1).masked_fill(~(band & m[:, :, None] & m[:, None]), 10000.0)
        coverage = d.min(1).values.clamp_min(1e-06).sqrt()[m]
        loss += 0.6 * coverage.mean() + 0.25 * torch.quantile(coverage, 0.9)
        return loss

    def validate():
        model.eval()
        with torch.no_grad():
            pred, logits, candidates, chosen = forward(val_index, True)
            m = mask[val_index]
            error = (pred - y[val_index]).square().sum(-1).sqrt()[m]
            dp = pred[:, 5:] - pred[:, :-5]
            dg = y[val_index, 5:] - y[val_index, :-5]
            dm = m[:, 5:] & m[:, :-5]
            displacement = (dp - dg).square().sum(-1).sqrt()[dm].mean()
            distance = (pred[:, :, None] - y[val_index, None]).square().sum(-1)
            band = (x['times'][val_index, :, None] - x['times'][val_index, None, :]).abs() <= 1.0
            coverage = distance.masked_fill(~(band & m[:, :, None] & m[:, None]), 10000.0).min(1).values.clamp_min(1e-06).sqrt()[m]
            row = dict(median_m=float(error.median()), mean_m=float(error.mean()), p90_m=float(torch.quantile(error, 0.9)), p95_m=float(torch.quantile(error, 0.95)), displacement_error_m=float(displacement), coverage_mean_m=float(coverage.mean()), coverage_p90_m=float(torch.quantile(coverage, 0.9)), direction_accuracy=float((logits.argmax(-1)[m] == state[val_index][m]).float().mean()), shape_window_s=float(model.shape_time_log.exp().clamp(0.5, 8.0)))
            row['selection_score'] = row['mean_m'] + 0.25 * row['p95_m'] + 0.4 * row['displacement_error_m'] + 0.6 * row['coverage_mean_m'] + 0.25 * row['coverage_p90_m']
            return row
    began = time.monotonic()
    best = float('inf')
    trace = []
    for update in range(1001):
        if update % 50 == 0:
            row = validate()
            row.update(update=update, seconds=time.monotonic() - began)
            trace.append(row)
            if row['selection_score'] < best:
                best = row['selection_score']
                torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, sensor_dim=arrays['sensor'].shape[-1], transition_generator=q, source_selection=row, source_training_devices=['oppo2', 'oppo3', 'maprealme', 'vivo'], target_labels_used=False, offline=True), a.output / 'trained.pt')
            print(json.dumps(row), flush=True)
        if update == 1000:
            break
        model.train()
        index = train_index[torch.randint(len(training), (128,), device=device)]
        optimizer.zero_grad(set_to_none=True)
        pred, logits, candidates, chosen = forward(index)
        loss = losses(pred, logits, candidates, chosen, index)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite source motion loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
    write(a.output / 'training_complete.json', dict(status='COMPLETE_SOURCE_ONLY_MOTION_MIXTURE_TRAINING', train_source_trajectories=len(training), heldout_source_trajectories=len(holdout), updates=1000, batch_size=128, new_parameters=sum((p.numel() for p in model.parameters())), source_selection_uses_MAP_path=True, target_labels_opened=False, trace=trace, seconds=time.monotonic() - began, GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated()))

def predict(a):
    torch.set_num_threads(1)
    with np.load(a.output / 'inputs.npz', allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    entries = json.loads((a.output / 'feature_manifest.json').read_text())['entries']
    saved = torch.load(a.output / 'trained.pt', map_location='cpu', weights_only=False)
    model = DirectionHead(saved['sensor_dim']).eval()
    model.load_state_dict(saved['state_dict'])
    transition = np.stack([expm(saved['transition_generator'] * float(arrays['times'][i, 1])) for i in range(len(entries))])
    from support_label_barrier import deny_label_reads
    root = a.output / 'predictions'
    began = time.monotonic()
    with deny_label_reads() as audit, torch.no_grad():
        x = {k: torch.from_numpy(v) for k, v in arrays.items()}
        logits, candidates = model(x['radio'], x['sensor'], x['lengths'])
        states = viterbi(logits, transition, arrays['lengths'])
        velocity = torch.gather(candidates, -2, states[..., None, None].expand(-1, -1, 1, 2)).squeeze(-2)
        result = model.fuse(velocity, x['base'], x['times'], x['lengths']).numpy()
        for i, e in enumerate(entries):
            c, s = (e['cohort'], e['segment'])
            source = a.capture / c / s['trajectory_id']
            folder = root / c / s['trajectory_id']
            folder.mkdir(parents=True, exist_ok=True)
            original = json.loads((source / 'predictions.json').read_text())
            t = np.asarray([(r['timestamp_ns'] - s['start_ns']) / 1000000000.0 for r in original])
            n = arrays['lengths'][i]
            xy = np.stack([np.interp(t, arrays['times'][i, :n], result[i, :n, j]) for j in range(2)], -1)
            predictions = [dict(p, xy_m=q.tolist(), inference_timing='OFFLINE_SOURCE_TRAINED_MAP_DIRECTIONS', later_position_GT_used=False) for p, q in zip(original, xy)]
            write(folder / 'predictions.json', predictions)
            np.savez_compressed(folder / 'motion_decode.npz', times_s=arrays['times'][i, :n], states=states[i, :n].numpy(), velocity_m_s=velocity[i, :n].numpy(), logits=logits[i, :n].numpy())
            for name in ['PRE_forecasts.npz', 'CS_diagnostics.json', 'WiFi_quality_diagnostics.json']:
                shutil.copyfile(str(source / name), str(folder / name))
        assert not audit['blocked_label_attempts']
    write(root / 'prediction_complete.json', dict(status='COMPLETE_SENSOR_ONLY_MAP_MOTION_INFERENCE', trajectories=len(entries), later_GT=False, offline_future_sensors=True, all_original_IDs_retained=True, seconds=time.monotonic() - began))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['prepare', 'train', 'predict'], required=True)
    for name in ['base', 'capture', 'features', 'proposal', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--lock', type=Path)
    a = p.parse_args()
    if a.stage == 'train':
        if not a.lock:
            raise ValueError('GPU execution lock required')
        with a.lock.open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(a)
    elif a.stage == 'prepare':
        prepare(a)
    else:
        predict(a)
if __name__ == '__main__':
    main()
