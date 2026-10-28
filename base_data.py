from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from base_event_training import EventSamples
TRAIN_DEVICES = ('oppo2', 'oppo3', 'maprealme', 'vivo')

def radio_tokens(rssi, mask, times):
    previous = np.r_[np.zeros_like(rssi[:1]), rssi[:-1]]
    prev_mask = np.r_[np.zeros_like(mask[:1]), mask[:-1]]
    dt = np.r_[0.0, np.diff(times)]
    delta = np.where(mask & prev_mask, rssi - previous, 0.0)
    age = np.zeros_like(rssi)
    last = np.full(rssi.shape[-1], -1000000000.0)
    for i in range(len(rssi)):
        age[i] = np.minimum(30.0, times[i] - last)
        last = np.where(mask[i], times[i], last)
    token = np.stack((rssi / 100, delta / 20, abs(delta) / 20, np.broadcast_to(dt[:, None] / 5, rssi.shape), age / 10, mask, prev_mask, np.broadcast_to(((dt > 0) & (dt < 10))[:, None], rssi.shape)), -1)
    return token.astype(np.float32)

def cs_snapshot_features(events, times_ns, anchors):
    history = defaultdict(list)
    features = np.zeros((len(times_ns), len(anchors), 20), np.float32)
    distance = np.zeros((len(times_ns), len(anchors)), np.float32)
    age = np.full_like(distance, 1000.0)
    ids = np.full(distance.shape, '', dtype=object)
    rows = [e for e in events if e['kind'] == 'cs']
    cursor = 0
    for i, time_ns in enumerate(times_ns):
        while cursor < len(rows) and rows[cursor]['monotonic_timestamp_ns'] <= time_ns:
            event = rows[cursor]
            p = event['payload']
            key = p['anchor_id']
            if key in anchors:
                history[key].append((event['monotonic_timestamp_ns'] / 1000000000.0, float(p['raw_range_m']), float(p['range_m']), event['observation_id']))
            cursor += 1
        now = time_ns / 1000000000.0
        for a, key in enumerate(anchors):
            seq = history[key]
            if not seq:
                continue
            t, raw, corrected, oid = seq[-1]
            interval = t - seq[-2][0] if len(seq) > 1 else 0.0
            valid_dt = 0.0001 < interval < 10
            change = corrected - seq[-2][2] if valid_dt else 0.0
            recent = [v for v in seq[-9:] if t - v[0] <= 10]
            values = np.asarray([v[2] for v in recent])
            median = np.median(values)
            onehot = [float(a == j) for j in range(len(anchors))]
            vector = [raw / 20, corrected / 20, (corrected - raw) / 10, change / 10, abs(change) / 10, interval / 5, change / max(interval, 0.01) / 10, median / 20, np.median(abs(values - median)) / 5, len(recent) / 9, (recent[-1][0] - recent[0][0]) / 10, min(now - t, 30) / 10, 1.0, float(len(seq) > 1), float(valid_dt), float(now - t <= 1.5), *onehot]
            assert len(vector) == 20
            features[i, a] = vector
            distance[i, a] = corrected
            age[i, a] = now - t
            ids[i, a] = oid
    return (features, distance, age, ids)

class BootstrapEpisodes:

    def __init__(self, base, data_root, split_root, capture, reference, anchors, cache):
        self.entries = []
        self.by_device = defaultdict(list)
        manifest = json.loads((Path(split_root) / 'train.json').read_text())
        if set(manifest['devices']) != set(TRAIN_DEVICES):
            raise ValueError('Training phone split differs')
        sessions = {}
        actor_field = None
        self.initial_recognition = 'first'
        actor_path = Path(reference).parent / 'actor.pt'
        if actor_path.exists():
            from base_model import PhysicalRadioField
            actor = torch.load(actor_path, map_location='cpu', weights_only=False)
            self.initial_recognition = actor['architecture'].get('initial_recognition', 'first')
            state = actor['state_dict']
            actor_field = PhysicalRadioField(cache, {key: state['field.' + key].tolist() for key in ['centres_m', 'exponents']}, actor['architecture'].get('orientation_basis') == 'relative_AP_harmonics', actor['architecture'].get('ap_basis_prior', False), actor['architecture'].get('fixed_ap_basis', False), actor['architecture'].get('source_field', 'idw'), actor['architecture'].get('source_inducing', 32), actor['architecture'].get('source_lengthscale_m', 1.2))
            actor_field.load_state_dict({key[len('field.'):]: value for key, value in state.items() if key.startswith('field.')})
            actor_field.eval()
        for segment in manifest['segments']:
            if not segment['full_length']:
                continue
            name, tid = (segment['device'], segment['trajectory_id'])
            if name not in TRAIN_DEVICES:
                raise ValueError('Validation/test phone in optimizer inputs')
            if name not in sessions:
                folder = Path(data_root) / name
                events = [json.loads(s) for s in (folder / 'online.ndjson').read_text().splitlines()]
                labels = json.loads((folder / 'evaluation_labels.json').read_text())
                sessions[name] = (events, labels)
            all_events, labels = sessions[name]
            events = [e for e in all_events if segment['start_ns'] <= e['monotonic_timestamp_ns'] < segment['end_ns']]
            folder = Path(capture) / tid / 'configured'
            with np.load(folder / 'rolling_pcrt_capture.npz', allow_pickle=False) as saved:
                rssi = saved['rssi'].copy()
                mask = saved['mask'].copy()
                times = saved['times_ns'].copy()
                motion = saved['motion_tokens'].copy()
                ids = saved['observation_ids'].copy()
                exact_motion = saved['motion_history'].copy() if 'motion_history' in saved.files else None
                exact_cs_age = saved['cs_age'].copy() if 'cs_age' in saved.files else None
            with np.load(Path(reference) / tid / 'native_reference_full25.npz', allow_pickle=False) as saved:
                if not np.array_equal(saved['observation_ids'], ids):
                    raise ValueError('CS/IMU reference does not match Wi-Fi event IDs')
                means = saved['means'][..., :2].astype(np.float32)
                headings = saved['means'][..., 4].astype(np.float32)
                cov = saved['covariances'][..., :2, :2].astype(np.float32)
                weights = saved['weights'].astype(np.float32)
            targets = np.asarray([labels.get(str(oid), {}).get('gt_xy_m', [np.nan, np.nan]) for oid in ids], np.float32)
            cs, ranges, ages, cs_ids = cs_snapshot_features(events, times, anchors)
            n = len(ids)
            if n < 18:
                continue
            elapsed = (times - times[0]) / 1000000000.0
            point = means.reshape(-1, 2)
            d2 = ((point[:, None] - cache['xy']) ** 2).sum(-1).clip(1e-05)
            w = d2[..., None] ** (-1) * cache['mask']
            source = ((w * (cache['trend'] + cache['residual'])).sum(1) / w.sum(1).clip(1e-08)).reshape(n, 8, -1)
            source = (source * weights[..., None]).sum(1).astype(np.float32)
            centroid = (means * weights[..., None]).sum(1)
            entry = dict(name=name, tid=tid, rssi=rssi, mask=mask, times=elapsed, radio=radio_tokens(rssi, mask, elapsed), motion=motion, means=means, covariance=cov, weights=weights, targets=targets, cs=cs, ranges=ranges, cs_age=ages, cs_ids=cs_ids, initial=segment['initial_position_xy_m'])
            entry.update(source=source, centroid=centroid, headings=headings)
            if exact_motion is not None:
                entry['motion_history'] = exact_motion
            if exact_cs_age is not None:
                entry['cs_age'] = exact_cs_age
            quality_path = Path(reference) / tid / 'WiFi_quality_diagnostics.json'
            if quality_path.exists():
                quality = json.loads(quality_path.read_text())
                if [value['observation_id'] for value in quality] != ids.tolist():
                    raise ValueError('Radio responsibilities do not match sensor IDs')
                entry['radio_nominal_responsibility'] = np.asarray([value['nominal_responsibility'] for value in quality], np.float32)
                entry['first_nominal_responsibility'] = entry['radio_nominal_responsibility'][0]
            prediction_path = Path(reference) / tid / 'predictions.json'
            starts = [0]
            if actor_field is not None and prediction_path.exists():
                predicted = json.loads(prediction_path.read_text())
                if [p['observation_id'] for p in predicted] != ids.tolist():
                    raise ValueError('Native region predictions do not match sensor IDs')
                origin = np.asarray(segment['initial_position_xy_m'])
                count = 0
                for i, prediction in enumerate(predicted):
                    if i:
                        count += 1
                    position = np.asarray(prediction['xy_m'])
                    displacement = float(np.linalg.norm(position - origin))
                    if count >= 4 and displacement >= 1.5 or (count >= 12 and displacement >= 0.6):
                        if i + 1 < n:
                            starts.append(i + 1)
                        origin = position
                        count = 0
            else:
                for i in range(1, n):
                    count = i - starts[-1]
                    displacement = float(np.linalg.norm(centroid[i] - centroid[starts[-1]]))
                    if count >= 4 and displacement >= 1.5 or (count >= 12 and displacement >= 0.6):
                        starts.append(i)
            entry['region_boundaries'] = np.asarray(starts + [n], dtype=np.int64)
            forecast_path = Path(reference) / tid / 'PRE_forecasts.npz'
            if forecast_path.exists() and actor_field is not None:
                with np.load(forecast_path, allow_pickle=False) as saved:
                    pm = saved['device_mean'].copy()
                    pc = saved['device_covariance'].copy()
                    context = saved['motion'].copy()
                    exact_innovation = saved['causal_innovation'].copy() if 'causal_innovation' in saved.files else None
                    if 'causal_device_design' in saved.files:
                        entry['native_device_design'] = saved['causal_device_design'].copy()
                    if 'causal_motion_path' in saved.files:
                        entry['native_motion_path'] = saved['causal_motion_path'].copy()
                    if 'transfer_mean' in saved.files:
                        entry.update({key: saved[key].copy() for key in ['transfer_mean', 'transfer_covariance', 'transfer_offset']})
                    if 'prefix_region_information' in saved.files:
                        entry.update({key: saved[key].copy() for key in ['prefix_region_information', 'prefix_region_vector']})
                    if 'prefix_closed_information' in saved.files:
                        entry.update({key: saved[key].copy() for key in ['prefix_closed_information', 'prefix_closed_vector', 'prefix_closed_probability', 'prefix_retired_information', 'prefix_retired_vector']})
                with torch.no_grad():
                    p = torch.tensor(means)
                    c = torch.tensor(context)[:, None].expand(n, 8, 32)
                    field = actor_field(p, c, heading=torch.tensor(headings))
                    uncalibrated = field['mean'] + torch.einsum('nkad,nkd->nka', field['basis'], torch.tensor(pm))
                entry.update(prefix_device_mean=pm, prefix_device_covariance=pc, native_innovations=exact_innovation if exact_innovation is not None else rssi[:, None] - uncalibrated.numpy())
            event_path = Path(reference) / tid / 'CS_training_contexts.npz'
            if event_path.exists():
                with np.load(event_path, allow_pickle=False) as saved:
                    events_cs = {key: saved[key].copy() for key in saved.files}
                true_range = []
                known = []
                anchor_xy = np.asarray(list(anchors.values()))
                for oid, anchor in zip(events_cs['observation_ids'], events_cs['anchor_index']):
                    label = labels.get(str(oid))
                    known.append(label is not None)
                    true_range.append(0.0 if label is None else float(np.linalg.norm(np.asarray(label['gt_xy_m']) - anchor_xy[int(anchor)])))
                events_cs.update(target_range=np.asarray(true_range, np.float32), label_mask=np.asarray(known, bool))
                entry['cs_events'] = events_cs
                if exact_cs_age is None:
                    ordered_age = np.full((n, len(anchors)), 1000.0, np.float32)
                    for time_ns, anchor, wifi_index in zip(events_cs['timestamp_ns'], events_cs['anchor_index'], events_cs['wifi_index']):
                        begin = max(0, int(wifi_index) + 1)
                        ordered_age[begin:, int(anchor)] = (times[begin:] - time_ns) / 1000000000.0
                    entry['cs_age'] = ordered_age
            self.by_device[name].append(len(self.entries))
            self.entries.append(entry)
        self.summary = dict(trajectories=len(self.entries), devices=list(TRAIN_DEVICES), source_only=True, geometry='Saved CS/IMU-only native reference; no Wi-Fi position input', labels='Separate loss-only coordinates', query_count=sum((len(e['rssi']) for e in self.entries)), warm_start_only=True, final_acceptance_requires_current_model_native_replay=True)
        self.summary.update(region_selection='Causal predicted displacement; no GT/time-only spatial proxy', region_count_per_trajectory=[len(e['region_boundaries']) - 1 for e in self.entries])
        if actor_field is not None:
            self.summary.update(geometry='Current three-module actor native PRE trajectories', forecast_innovations='Actual past uncorrected PRE predictions, no GT coordinates', actor_checkpoint=str(actor_path), initial_prior='Immutable causal actor device cavity at window start', warm_start_only=False, training_mode='Joint conditional windows with captured native prefixes; no full-trajectory BPTT')
            self.summary.update(actor_mode_evidence=actor['architecture'].get('mode_evidence', 'local'), actor_initial_posterior=actor['architecture'].get('initial_posterior', 'none'), forecast_innovations='Exact captured conditional nuisance innovations' if actor['architecture'].get('innovation_memory') == 'conditional_nuisance' else 'Actual past uncorrected PRE predictions, no GT coordinates')
        self.event_source = EventSamples(self.entries)
        self.summary.update(self.event_source.summary)

    def sample(self, rng, batch=128, regions=4, scans=8, history=16, device='cuda'):
        output = defaultdict(list)
        for b in range(batch):
            phone = TRAIN_DEVICES[int(rng.integers(len(TRAIN_DEVICES)))]
            entry = self.entries[int(rng.choice(self.by_device[phone]))]
            n = len(entry['rssi'])
            boundaries = entry['region_boundaries']
            choices = [j for j in range(len(boundaries) - 1) if boundaries[j + 1] > 5]
            first_region = int(rng.choice(choices))
            begin = max(5, int(boundaries[first_region]))
            indices = []
            for offset in range(regions):
                group = first_region + offset
                if group >= len(boundaries) - 1:
                    indices.append((np.full(scans, n - 1, dtype=np.int64), np.zeros(scans, bool)))
                    continue
                l = max(begin, int(boundaries[group]))
                r = int(boundaries[group + 1])
                ix = np.arange(l, min(r, l + scans))
                valid = np.ones(len(ix), bool)
                if len(ix) < scans:
                    valid = np.r_[valid, np.zeros(scans - len(ix), bool)]
                    ix = np.pad(ix, (0, scans - len(ix)), constant_values=max(l, r - 1))
                indices.append((ix, valid))
            idx = np.stack([v[0] for v in indices])
            valid = np.stack([v[1] for v in indices])
            for key in ['rssi', 'mask', 'motion', 'means', 'headings', 'covariance', 'weights', 'targets', 'cs', 'ranges', 'cs_age']:
                output[key].append(entry[key][idx])
            if 'radio_nominal_responsibility' in entry:
                output['radio_nominal_responsibility'].append(entry['radio_nominal_responsibility'][idx])
            if 'transfer_mean' in entry:
                for key in ['transfer_mean', 'transfer_covariance', 'transfer_offset']:
                    output[key].append(entry[key][idx])
            if 'prefix_region_information' in entry:
                for key in ['prefix_region_information', 'prefix_region_vector']:
                    output[key].append(entry[key][begin])
                output['first_means'].append(entry['means'][0])
                output['first_covariance'].append(entry['covariance'][0])
                output['first_headings'].append(entry['headings'][0])
                output['first_motion_history'].append(entry['motion_history'][0])
                output['first_nominal_responsibility'].append(entry.get('first_nominal_responsibility', np.ones(8, np.float32)))
            output['mask'][-1] &= valid[..., None]
            output['valid'].append(valid & np.isfinite(entry['targets'][idx]).all(-1))
            output['initial_xy'].append(entry['initial'])
            output['initial_rssi'].append(entry['rssi'][0])
            output['initial_mask'].append(entry['mask'][0])
            if 'prefix_device_mean' in entry:
                output['prefix_device_mean'].append(entry['prefix_device_mean'][begin])
                output['prefix_device_covariance'].append(entry['prefix_device_covariance'][begin])
            histories = []
            initial = []
            mh = []
            source_h = []
            age_h = []
            spatial_h = []
            rotation_h = []
            innovation_h = []
            for region in range(regions):
                rh = []
                ih = []
                mm = []
                sh = []
                ah = []
                ph = []
                rt = []
                iv = []
                for index in idx[region]:
                    lo = max(0, index - history)
                    h = entry['radio'][lo:index]
                    h = np.pad(h, ((history - len(h), 0), (0, 0), (0, 0)))
                    rh.append(h.transpose(1, 0, 2))
                    sh.append(np.pad(entry['source'][lo:index], ((history - (index - lo), 0), (0, 0))))
                    age = entry['times'][index] - entry['times'][lo:index]
                    ah.append(np.pad(age, (history - len(age), 0), constant_values=1000.0))
                    displacement = ((entry['means'][lo:index] - entry['means'][index]) ** 2).sum(-1)
                    current_var = np.trace(entry['covariance'][index], axis1=-2, axis2=-1)
                    previous_var = np.trace(entry['covariance'][lo:index], axis1=-2, axis2=-1)
                    spatial = np.exp(-0.5 * displacement / (1 + current_var[None] + previous_var))
                    ph.append(np.pad(spatial, ((history - len(spatial), 0), (0, 0))).T)
                    axis = entry['motion'][lo:index][:, [0, 2]]
                    current = entry['motion'][index, [0, 2]]
                    cosine = (axis * current).sum(-1) / (np.linalg.norm(axis, axis=-1) * np.linalg.norm(current)).clip(1e-06)
                    rotation = np.exp(2 * (cosine.clip(-1, 1) - 1))
                    rt.append(np.pad(rotation, (history - len(rotation), 0)))
                    if 'native_innovations' in entry:
                        innovations = entry['native_innovations'][lo:index]
                        iv.append(np.pad(innovations, ((history - len(innovations), 0), (0, 0), (0, 0))).transpose(1, 2, 0))
                    first = entry['radio'][:5]
                    ih.append(first.transpose(1, 0, 2))
                    if 'motion_history' in entry:
                        mm.append(entry['motion_history'][index])
                    else:
                        m = entry['motion'][max(0, index - history + 1):index + 1]
                        mm.append(np.pad(m, ((history - len(m), 0), (0, 0))))
                histories.append(rh)
                initial.append(ih)
                mh.append(mm)
                source_h.append(sh)
                age_h.append(ah)
                spatial_h.append(ph)
                rotation_h.append(rt)
                if iv:
                    innovation_h.append(iv)
            output['radio_history'].append(histories)
            output['initial_history'].append(initial)
            output['motion_history'].append(mh)
            output['source_history'].append(source_h)
            output['age_history'].append(age_h)
            output['spatial_history'].append(spatial_h)
            output['rotation_history'].append(rotation_h)
            if innovation_h:
                output['native_innovation_history'].append(innovation_h)
        result = {k: torch.as_tensor(np.asarray(v), device=device) for k, v in output.items()}
        if self.event_source.phones:
            result.update({k: torch.as_tensor(v, device=device) for k, v in self.event_source.sample(rng, batch * 4, history).items()})
        return {k: v.float() if v.dtype == torch.float64 else v for k, v in result.items()}
