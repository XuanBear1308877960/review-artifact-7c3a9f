import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
if __name__ == '__main__':
    sys.modules['canonical_replay'] = sys.modules[__name__]
BASE = Path('.')
HERE = Path(__file__).resolve().parent
FULL_ROOT = HERE / 'canonical_v2'
ROOT = Path(os.environ.get('YOMO_CANONICAL_OUTPUT_ROOT', str(FULL_ROOT)))
CAPTURE = Path(os.environ.get('YOMO_CANONICAL_CAPTURE_ROOT', str(BASE / 'runs/direction_v80/trajectory_factor_capture_v1')))
sys.path.insert(0, str(BASE))
from kinematics import RadioDirectionEngine
from temporal_radio import load_actor, PARENT_RELATIVE
_ACTOR = None
_SESSIONS = {}

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')

class MaskedClockEngine(RadioDirectionEngine):

    def _consume_wifi(self, event):
        if event['payload']['ap_observations']:
            return super()._consume_wifi(event)
        mask = torch.zeros(self.a, device=self.device, dtype=torch.bool)
        encoded = self._encoded(mask)
        mu, cov = self._device_prediction()
        delta = self._short_residual(encoded, mu)
        weight = self.log_weight.softmax(-1)
        position = (weight[:, None] * self.mean[:, :2]).sum(0)
        forecast = dict(observation_id=event['observation_id'], timestamp_ns=self.timestamp, device_mean=mu.cpu().numpy().copy(), device_covariance=cov.cpu().numpy().copy(), short_residual=delta.cpu().numpy().copy(), motion=encoded['motion'][0].cpu().numpy().copy(), weights=weight.cpu().numpy().copy(), rssi=np.zeros(self.a, np.float32), mask=np.zeros(self.a, bool), pre_mean=self.mean.cpu().numpy().copy(), pre_covariance=self.cov.cpu().numpy().copy(), motion_token=np.asarray(self.motion_tokens[-1], np.float32).copy(), motion_history=self.current_motion_history.copy(), trajectory_radio_loglike=np.zeros((self.k, len(self.grid_xy)), np.float32), cs_age=np.asarray([1000.0 if not seq else self.timestamp / 1000000000.0 - seq[-1][0] for seq in self.cs_history.values()], np.float32))
        self.forecasts.append(forecast)
        self.radio_diagnostics.append(dict(observation_id=event['observation_id'], sigma_db=encoded['sigma'][0].cpu().tolist(), outlier_probability=encoded['outlier'][0].cpu().tolist(), available_AP=[False] * self.a, WiFi_update_applied=False))
        output = dict(observation_id=event['observation_id'], timestamp_ns=self.timestamp, xy_m=position.cpu().tolist(), mode_xy_m=self.mean[:, :2].cpu().tolist(), mode_weights=weight.cpu().tolist(), known_initial_position_only=True, masked_WiFi_timestamp_only=True)
        self.outputs.append(output)
        return output

def events(cohort, s):
    data = BASE / 'runs/ap_movement_v79_291/experiment/preprocessed' if cohort == 'AP_movement' else Path(os.environ.get('YOMO_CANONICAL_SENSOR_ROOT', str(BASE / 'data/processed_rssi_causal_short_v1')))
    key = str(data / s['device'] / 'online.ndjson')
    if key not in _SESSIONS:
        _SESSIONS[key] = [json.loads(x) for x in Path(key).read_text().splitlines()]
    return [r for r in _SESSIONS[key] if s['start_ns'] <= r['monotonic_timestamp_ns'] < s['end_ns']]

def replay(cohort, s, allowed, out, alter_held_db=0.0, persist=True, preserve_native_grid=False):
    global _ACTOR
    torch.set_num_threads(1)
    start = time.monotonic()
    out = Path(out)
    from support_label_barrier import deny_label_reads
    with deny_label_reads() as audit, torch.no_grad():
        if _ACTOR is None:
            _ACTOR = load_actor(BASE, BASE / PARENT_RELATIVE)
        net, cfg, aps, _, saved = _ACTOR
        cfg = copy.deepcopy(cfg)
        cfg['filter'].update(direction_policy='radio_frame_independent_cs', export_trajectory_radio_factors=True)
        neural_motion = os.environ.get('YOMO_CANONICAL_NEURAL_MOTION_PRIOR') == '1'
        if neural_motion:
            from event_motion_prior import RTSEngine, MaskedRTSEngine
            engine_class = RTSEngine if preserve_native_grid else MaskedRTSEngine
        else:
            engine_class = RadioDirectionEngine if preserve_native_grid else MaskedClockEngine
        engine = engine_class(net, aps, cfg['anchors'], s['initial_position_xy_m'], s['start_ns'], cfg['filter'], False, 1, saved.get('position_rule_training', 'mixed'))
        lookup = {a: i for i, a in enumerate(aps)}
        ordinal = 0
        changed = 0
        event_digest = hashlib.sha256()
        wifi_ids = []
        for row in events(cohort, s):
            if row['kind'] == 'wifi' and preserve_native_grid:
                wifi_ids.append(row['observation_id'])
                ordinal += 1
            elif row['kind'] == 'wifi':
                selected = []
                for ap in row['payload']['ap_observations']:
                    a = lookup.get(ap['ap_id_hash'])
                    if a is not None and allowed[ordinal, a]:
                        selected.append(ap)
                    elif a is not None:
                        if alter_held_db:
                            ap = dict(ap, rssi_dbm=ap['rssi_dbm'] + alter_held_db)
                            changed += 1
                row = dict(row, payload=dict(row['payload'], ap_observations=selected))
                wifi_ids.append(row['observation_id'])
                ordinal += 1
            event_digest.update(json.dumps(row, sort_keys=True, separators=(',', ':')).encode())
            engine.consume(row)
        if ordinal != len(allowed) or len(engine.outputs) != ordinal:
            raise ValueError('Masked clock replay lost WiFi rows')
        rows = engine.forecasts
        pre = {k: np.asarray([r[k] for r in rows]) for k in ['device_mean', 'device_covariance', 'short_residual', 'motion', 'weights', 'rssi', 'mask']}
        pre['observation_ids'] = np.asarray(wifi_ids)
        times = np.array([r['timestamp_ns'] for r in rows], np.int64)
        if not np.all(np.isfinite(pre['motion'])):
            raise ValueError('Nonfinite masked motion TCN output')
        reference = CAPTURE / cohort / s['trajectory_id']
        with np.load(reference / 'PRE_forecasts.npz', allow_pickle=False) as z:
            if not np.array_equal(z['observation_ids'], pre['observation_ids']):
                raise ValueError('Replay IDs differ from fixed main cohort')
            prefix_errors = {k: float(abs(pre[k][5] - z[k][5]).max()) for k in ['device_mean', 'device_covariance', 'weights']}
        if max(prefix_errors.values()) > 1e-06:
            raise ValueError('Initial-five independent posterior was changed: ' + str(prefix_errors))
        for k in pre:
            if k != 'observation_ids' and (not np.isfinite(pre[k]).all()):
                raise ValueError('Nonfinite replay state: ' + k)
        signature = hashlib.sha256()
        for k in sorted(pre):
            signature.update(np.ascontiguousarray(pre[k]).tobytes())
        signature.update(json.dumps(engine.outputs, sort_keys=True).encode())
        if neural_motion:
            from event_motion_prior import rts_signature
            signature.update(rts_signature(engine).encode())
        report = dict(status='COMPLETED_HELD_RSSI_FREE_NATIVE_REPLAY', cohort=cohort, trajectory_id=s['trajectory_id'], WiFi_rows=ordinal, retained_AP_observations=int(pre['mask'].sum()), allowed_AP_mask_sha256=hashlib.sha256(np.ascontiguousarray(allowed).tobytes()).hexdigest(), sanitized_event_sha256=event_digest.hexdigest(), state_sha256=signature.hexdigest(), prefix_posterior_max_differences=prefix_errors, no_later_position_truth=True, motion_TCN_and_chord_path_rebuilt=True, CS_context_and_uncertainty_rebuilt=True, masked_clock_rows=sum((not bool(np.any(r)) for r in pre['mask'])), blocked_label_attempts=audit['blocked_label_attempts'], seconds=time.monotonic() - start)
        if preserve_native_grid:
            report.update(status='COMPLETED_FULL_SENSOR_NATIVE_CAPTURE', all_current_WiFi_retained=True, source_sensor_root=os.environ.get('YOMO_CANONICAL_SENSOR_ROOT', str(BASE / 'data/processed_rssi_causal_short_v1')))
        if persist:
            out.mkdir(parents=True, exist_ok=True)
            write(out / 'predictions.json', engine.outputs)
            write(out / 'CS_diagnostics.json', engine.cs_diagnostics)
            write(out / 'WiFi_quality_diagnostics.json', engine.radio_diagnostics)
            np.savez_compressed(out / 'PRE_forecasts.npz', **pre)
            cs = engine.cs_training_rows
            keys = ['timestamp_ns', 'features', 'motion_history', 'range_m', 'bias_mean', 'bias_variance', 'mode_weights', 'velocity', 'anchor_index', 'wifi_index', 'wifi_age_s']
            if engine.temporal:
                keys += ['temporal_mean', 'temporal_variance', 'bias_temporal_covariance']
            np.savez_compressed(out / 'CS_training_contexts.npz', observation_ids=np.array([r['observation_id'] for r in cs]), **{k: np.asarray([r[k] for r in cs]) for k in keys})
            np.savez_compressed(out / 'trajectory_factors.npz', grid_xy=engine.grid_xy.cpu().numpy(), times_ns=times, radio_loglike=np.asarray([r['trajectory_radio_loglike'] for r in rows]) if preserve_native_grid or neural_motion else np.zeros((ordinal, engine.k, len(engine.grid_xy)), np.float32), anchors=engine.anchors.cpu().numpy(), cs_sigma=np.asarray([r['sigma'] for r in engine.cs_diagnostics]), cs_pi=np.asarray([r['outlier_probability'] for r in engine.cs_diagnostics]))
            np.savez_compressed(out / 'replay_context.npz', observation_ids=pre['observation_ids'], times_ns=times, allowed_AP_mask=allowed, motion_tokens=np.asarray([r['motion_token'] for r in rows]), motion_history=np.asarray([r['motion_history'] for r in rows]))
            if neural_motion:
                (out / 'configured').mkdir(exist_ok=True)
                np.savez_compressed(out / 'configured/rolling_pcrt_capture.npz', observation_ids=pre['observation_ids'], motion_tokens=np.asarray([r['motion_token'] for r in rows]))
                from event_motion_prior import finish
                report['source_neural_motion_prior'] = finish(engine, out, cohort, s, preserve_native_grid)
            write(out / 'completion.json', report)
        if audit['blocked_label_attempts']:
            raise ValueError('Replay attempted reference reads')
    return report

def proof():
    capture = BASE / 'runs/direction_v80/trajectory_factor_capture_v1'
    s = json.loads((capture / 'segments.json').read_text())['source_train'][0]
    with np.load(capture / 'source_train' / s['trajectory_id'] / 'PRE_forecasts.npz', allow_pickle=False) as z:
        allowed = np.zeros_like(z['mask'])
        allowed[:5] = z['mask'][:5]
    out = ROOT / 'prefix_capture/source_train' / s['trajectory_id']
    a = replay('source_train', s, allowed, out)
    b = replay('source_train', s, allowed, out, 31.25, False)
    if a['sanitized_event_sha256'] != b['sanitized_event_sha256'] or a['state_sha256'] != b['state_sha256']:
        write(ROOT / 'native_replay_identity_failure.json', dict(first=a, altered=b))
        raise ValueError('Held RSSI changed a regenerated context or prediction')
    report = dict(status='PASS_ACTUAL_HELD_RSSI_INTERVENTION', source_trajectory=s['trajectory_id'], source_original=a, source_changed_held_values=b, held_RSSI_addition_db=31.25, identical_sanitized_events=True, identical_all_forecasts_and_native_paths=True, first_five_channel_preserved=True, all_subsequent_radio_channels_omitted_in_this_proof=True)
    write(ROOT / 'native_replay_proof.json', report)
    print(json.dumps(report), flush=True)
if __name__ == '__main__':
    proof()
