import argparse
import json
from pathlib import Path
import numpy as np
from analysis_utils import interpolate, statistics
from engine_utils import score_cohort, write
from temporal_radio import PARENT_RELATIVE

def shape(folder, segments):
    rows = json.loads((folder / 'scored_predictions.json').read_text())
    per, angles, bad, pa, ga = ({}, [], [], [], [])
    for s in segments:
        tid = s['trajectory_id']
        pred = json.loads((folder / tid / 'predictions.json').read_text())
        lookup = {p['observation_id']: p for p in pred}
        selected = [r for r in rows if r['trajectory_id'] == tid]
        selected.sort(key=lambda r: lookup[r['observation_id']]['timestamp_ns'])
        t = np.asarray([(lookup[r['observation_id']]['timestamp_ns'] - s['start_ns']) / 1000000000.0 for r in selected])
        p = np.asarray([r['predicted_xy_m'] for r in selected])
        g = np.asarray([r['reference_xy_m'] for r in selected])
        eligible = t >= t[0] + 1.0
        dp = p[eligible] - interpolate(t, p, t[eligible] - 1.0)
        dg = g[eligible] - interpolate(t, g, t[eligible] - 1.0)
        moving = np.linalg.norm(dg, axis=-1) >= 0.5
        dp, dg = (dp[moving], dg[moving])
        undefined = np.linalg.norm(dp, axis=-1) < 0.15
        angle = abs(np.arctan2(dp[:, 0] * dg[:, 1] - dp[:, 1] * dg[:, 0], (dp * dg).sum(-1))) * 180 / np.pi
        b = (angle > 90) | undefined
        angles.extend(angle[~undefined].tolist())
        bad.extend(b.tolist())
        grid = np.arange(t[0], t[-1] + 1e-06, 1.0)
        parc = float(np.linalg.norm(np.diff(interpolate(t, p, grid), axis=0), axis=-1).sum())
        garc = float(np.linalg.norm(np.diff(interpolate(t, g, grid), axis=0), axis=-1).sum())
        pa.append(parc)
        ga.append(garc)
        dist = np.linalg.norm(p[:, None, :] - g[None, :, :], axis=-1)
        table = np.full_like(dist, np.inf)
        for i in range(len(p)):
            for j in range(len(g)):
                if i == j == 0:
                    previous = 0.0
                else:
                    previous = min(table[i - 1, j] if i else np.inf, table[i, j - 1] if j else np.inf, table[i - 1, j - 1] if i and j else np.inf)
                table[i, j] = max(dist[i, j], previous)
        per[tid] = dict(wrong_or_undefined_fraction=float(np.mean(b)), direction_angle_median_deg=float(np.median(angle[~undefined])), ordered_curve_distance_m=float(table[-1, -1]), predicted_1s_arc_m=parc, reference_1s_arc_m=garc, errors=statistics([r['error_m'] for r in selected]))
    return dict(direction_angle_deg=statistics(angles), wrong_or_undefined_fraction=float(np.mean(bad)), one_second_path_length_ratio=float(sum(pa) / sum(ga)), ordered_curve_distance_m=statistics([d['ordered_curve_distance_m'] for d in per.values()]), per_trajectory=per)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--cohorts', nargs='+', required=True)
    a = p.parse_args()
    if not (a.candidate / 'prediction_complete.json').exists():
        raise ValueError('Finish all sensor-only predictions before scoring')
    cohorts = json.loads((a.capture / 'segments.json').read_text())
    candidate_metadata = json.loads((a.candidate / 'prediction_complete.json').read_text())
    results = {}
    for tag, root in [('D01', a.capture), ('joint_sensor_path', a.candidate)]:
        results[tag] = {}
        for c in a.cohorts:
            data = a.base / ('runs/ap_movement_v79_291/experiment/preprocessed' if c == 'AP_movement' else 'data/processed_rssi_causal_short_v1')
            result = score_cohort(a.base, a.base / PARENT_RELATIVE, cohorts[c], root / c, data)
            if tag == 'joint_sensor_path' and candidate_metadata.get('offline_selected_calibration_branch', False):
                result['RSSI_scope'] = 'Offline trajectory-selected calibration branch; conditional PRE device parameters and residuals, evaluator GT XY only'
                write(root / c / 'complete.json', result)
            results[tag][c] = dict(pooled_metrics=result['pooled_metrics'], pooled_RSSI_MAE_db=result['pooled_RSSI_MAE_db'], shape=shape(root / c, cohorts[c]))
            print(json.dumps(dict(policy=tag, cohort=c, **{k: v for k, v in results[tag][c].items() if k != 'shape'}, shape={k: v for k, v in results[tag][c]['shape'].items() if k != 'per_trajectory'})), flush=True)
    write(a.candidate / 'comparison.json', dict(status='COMPLETE_TRAJECTORY_FACTOR_COMPARISON', results=results, source_fit_only=True, offline=True, later_position_GT_in_solver=False, scoring_exclusions_changed=False))
if __name__ == '__main__':
    main()
