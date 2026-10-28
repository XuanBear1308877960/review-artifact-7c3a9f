import argparse
import csv
import json
from pathlib import Path
import numpy as np

def statistics(values):
    a = np.asarray(values, float)
    if not len(a):
        return dict(n=0)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)), p90=float(np.quantile(a, 0.9)), p95=float(np.quantile(a, 0.95)))

def interpolate(times, xy, query):
    return np.stack([np.interp(query, times, xy[:, j]) for j in range(2)], -1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--segments', type=Path, required=True)
    args = parser.parse_args()
    complete = json.loads((args.root / 'complete.json').read_text())
    cohorts = json.loads(args.segments.read_text())
    results, records = ({}, [])
    for policy, values in complete['results'].items():
        results[policy] = {}
        for cohort in values:
            scores = json.loads((args.root / policy / cohort / 'scored_predictions.json').read_text())
            by_track = {}
            for row in scores:
                by_track.setdefault(row['trajectory_id'], []).append(row)
            all_angle, all_bad, all_undefined, predicted_arc, reference_arc, late, track_stats = ([], [], [], [], [], [], {})
            for segment in cohorts[cohort]:
                tid = segment['trajectory_id']
                rows = by_track[tid]
                predictions = json.loads((args.root / policy / cohort / tid / 'predictions.json').read_text())
                lookup = {r['observation_id']: r for r in predictions}
                rows.sort(key=lambda r: lookup[r['observation_id']]['timestamp_ns'])
                times = np.asarray([(lookup[r['observation_id']]['timestamp_ns'] - segment['start_ns']) / 1000000000.0 for r in rows])
                xy = np.asarray([r['predicted_xy_m'] for r in rows])
                gt = np.asarray([r['reference_xy_m'] for r in rows])
                eligible = times >= times[0] + 1.0
                delta_xy = xy[eligible] - interpolate(times, xy, times[eligible] - 1.0)
                delta_gt = gt[eligible] - interpolate(times, gt, times[eligible] - 1.0)
                moving = np.linalg.norm(delta_gt, axis=-1) >= 0.5
                predicted_length = np.linalg.norm(delta_xy[moving], axis=-1)
                undefined = predicted_length < 0.15
                cross = delta_xy[moving, 0] * delta_gt[moving, 1] - delta_xy[moving, 1] * delta_gt[moving, 0]
                dot = (delta_xy[moving] * delta_gt[moving]).sum(-1)
                angle = abs(np.arctan2(cross, dot)) * 180 / np.pi
                all_angle.extend(angle[~undefined].tolist())
                all_bad.extend(((angle > 90) | undefined).tolist())
                all_undefined.extend(undefined.tolist())
                grid = np.arange(times[0], times[-1] + 1e-06, 1.0)
                p = float(np.linalg.norm(np.diff(interpolate(times, xy, grid), axis=0), axis=-1).sum())
                g = float(np.linalg.norm(np.diff(interpolate(times, gt, grid), axis=0), axis=-1).sum())
                predicted_arc.append(p)
                reference_arc.append(g)
                errors = [r['error_m'] for r in rows]
                late.extend([r['error_m'] for t, r in zip(times, rows) if t >= 0.75 * segment['duration_s']])
                track_stats[tid] = dict(errors=statistics(errors), direction_error_deg=statistics(angle[~undefined]), wrong_or_undefined_fraction=float(np.mean((angle > 90) | undefined)), predicted_arc_m=p, reference_arc_m=g)
                for t, row in zip(times, rows):
                    records.append(dict(policy=policy, cohort=cohort, trajectory_id=tid, observation_id=row['observation_id'], elapsed_s=float(t), predicted_x_m=row['predicted_xy_m'][0], predicted_y_m=row['predicted_xy_m'][1], reference_x_m=row['reference_xy_m'][0], reference_y_m=row['reference_xy_m'][1], error_m=row['error_m']))
            results[policy][cohort] = dict(**values[cohort], direction_error_deg=statistics(all_angle), direction_interval_count=len(all_bad), wrong_or_undefined_direction_fraction=float(np.mean(all_bad)), undefined_direction_fraction=float(np.mean(all_undefined)), pooled_path_length_ratio=float(sum(predicted_arc) / sum(reference_arc)), total_predicted_arc_m=float(sum(predicted_arc)), total_reference_arc_m=float(sum(reference_arc)), last_quarter_error_m=statistics(late), per_trajectory=track_stats)
    result = dict(status='COMPLETE_NATIVE_LOCATION_AND_DIRECTION_ANALYSIS', results=results, scope='All original scoring IDs; movement direction diagnostic uses fixed one-second GT displacement>=0.5m, and reports predictions<0.15m as undefined rather than hiding them', no_GT_used_for_prediction=True, post_smoothing_applied=False)
    (args.root / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (args.root / 'point_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({p: {c: {k: v for k, v in d.items() if k != 'per_trajectory'} for c, d in a.items()} for p, a in results.items()}), flush=True)
if __name__ == '__main__':
    main()
