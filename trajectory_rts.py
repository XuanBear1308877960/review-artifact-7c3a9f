import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import base_evaluate as parent
from kinematics import RadioDirectionEngine
from temporal_radio import load_actor, PARENT_RELATIVE
from engine_utils import write
_CACHE = None

class RTSEngine(RadioDirectionEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rts_states = [(self.timestamp, self.mean.detach().cpu().numpy().copy(), self.cov.detach().cpu().numpy().copy())]
        self.rts_predictions = []

    def _advance(self, ns):
        dt = (int(ns) - self.timestamp) / 1000000000.0
        if dt <= 0:
            return super()._advance(ns)
        x = self.mean.detach().cpu().numpy().astype(float)
        p = self.cov.detach().cpu().numpy().astype(float)
        d = x.shape[-1]
        F = np.broadcast_to(np.eye(d), (self.k, d, d)).copy()
        rate = self.filter_cfg.get('velocity_relaxation_rate_hz', 0.0)
        decay = np.exp(-rate * dt) if rate > 1e-06 else 1.0
        vdt = -np.expm1(-rate * dt) / rate if rate > 1e-06 else dt
        adt = (dt - vdt) / rate if rate > 1e-06 else 0.5 * dt * dt
        F[:, 0, 2] = F[:, 1, 3] = vdt
        F[:, 2, 2] = F[:, 3, 3] = decay
        if self.motion.available:
            angle = x[:, 4]
            c, s = (np.cos(angle), np.sin(angle))
            ax, ay = self.motion.accel
            derivative = np.stack((-s * ax - c * ay, c * ax - s * ay), -1)
            F[:, :2, 4] = derivative * adt
            F[:, 2:4, 4] = derivative * vdt
        super()._advance(ns)
        self.rts_predictions.append((self.mean.detach().cpu().numpy().astype(float), self.cov.detach().cpu().numpy().astype(float), p @ F.swapaxes(-1, -2)))
        self.rts_states.append((self.timestamp, None, None))

    def consume(self, event):
        result = super().consume(event)
        self.rts_states[-1] = (self.timestamp, self.mean.detach().cpu().numpy().astype(float), self.cov.detach().cpu().numpy().astype(float))
        return result

    def backward_means(self):
        smooth = [s[1].copy() for s in self.rts_states]
        max_condition = 0.0
        for i in range(len(smooth) - 2, -1, -1):
            forecast, pcross, cross = self.rts_predictions[i]
            gain = np.linalg.solve(pcross.swapaxes(-1, -2), cross.swapaxes(-1, -2)).swapaxes(-1, -2)
            residual = smooth[i + 1] - forecast
            residual[:, 4] = (residual[:, 4] + np.pi) % (2 * np.pi) - np.pi
            smooth[i] = self.rts_states[i][1] + np.einsum('kde,ke->kd', gain, residual)
            if not np.isfinite(smooth[i]).all():
                raise FloatingPointError('Nonfinite full-state RTS mean')
        return {s[0]: x for s, x in zip(self.rts_states, smooth)}

def job(task):
    global _CACHE
    base, output, cohort, s = task
    base, output = (Path(base), Path(output))
    torch.set_num_threads(1)
    if parent.AVAILABLE_CPUS:
        os.sched_setaffinity(0, parent.AVAILABLE_CPUS)
    data = base / ('runs/ap_movement_v79_291/experiment/preprocessed' if cohort == 'AP_movement' else 'data/processed_rssi_causal_short_v1')
    from support_label_barrier import deny_label_reads
    with deny_label_reads() as audit:
        if _CACHE is None:
            _CACHE = load_actor(base, base / PARENT_RELATIVE)
        net, cfg, aps, _, saved = _CACHE
        cfg = copy.deepcopy(cfg)
        cfg['filter'].update(direction_policy='radio_frame_independent_cs')
        parent.NativeEngine = RTSEngine
        parent._ACTOR = (net, cfg, aps, data, saved)
        native = output / 'forward' / cohort
        parent.predict_trajectory((str(base), str(base / PARENT_RELATIVE), s, str(native), False, 1, None))
        engine = RadioDirectionEngine.last_instance
        means = engine.backward_means()
        weights = engine.log_weight.softmax(-1).cpu().numpy()
        selected = int(np.argmax(weights))
        for policy in ['rts_mean', 'rts_branch']:
            folder = output / policy / cohort / s['trajectory_id']
            folder.mkdir(parents=True, exist_ok=True)
            predictions = []
            for original in engine.outputs:
                x = means[original['timestamp_ns']]
                xy = (weights[:, None] * x[:, :2]).sum(0) if policy == 'rts_mean' else x[selected, :2]
                p = dict(original, xy_m=xy.tolist(), mode_xy_m=x[:, :2].tolist(), mode_weights=weights.tolist(), inference_timing='OFFLINE_FULL_STATE_RTS', later_position_GT_used=False)
                predictions.append(p)
            write(folder / 'predictions.json', predictions)
            for name in ['PRE_forecasts.npz', 'CS_diagnostics.json', 'WiFi_quality_diagnostics.json']:
                shutil.copyfile(str(native / s['trajectory_id'] / name), str(folder / name))
        assert not audit['blocked_label_attempts']
    return dict(trajectory_id=s['trajectory_id'], timeline_states=len(engine.rts_states), sensor_factors_counted_once=True, labels_opened=False, backward_mode_weights=weights.tolist())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--segments', type=Path, required=True)
    p.add_argument('--workers', type=int, default=24)
    a = p.parse_args()
    began = time.monotonic()
    cohorts = json.loads(a.segments.read_text())
    jobs = [(str(a.base), str(a.output), c, s) for c, segments in cohorts.items() for s in segments]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        reports = list(pool.map(job, jobs))
    for policy in ['rts_mean', 'rts_branch']:
        write(a.output / policy / 'prediction_complete.json', dict(status='COMPLETE_LABEL_GUARDED_FULL_STATE_RTS', reports=reports, offline_future_sensors_used=True, position_GT_used=False, seconds=time.monotonic() - began))
    print(json.dumps(dict(completed=len(reports), seconds=time.monotonic() - began)), flush=True)
if __name__ == '__main__':
    main()
