import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import base_evaluate as parent
from base_native import NativeEngine
from kinematics import RadioDirectionEngine
from temporal_radio import load_actor as load_direction_actor
_CACHE = None
LOAD_PARENT = load_direction_actor
POLICIES = ['control', 'radio_constant_velocity', 'radio_velocity_mixture', 'radio_frame_registration', 'radio_frame_bias_cv', 'radio_model_mixture', 'radio_frame_registration_aligned', 'radio_spatial_basin', 'radio_spatial_basin_cv', 'control_balanced', 'radio_balanced', 'radio_balanced_median', 'radio_temporal_velocity_mixture', 'radio_temporal_constant_velocity', 'radio_speed_magnitude', 'radio_velocity_centered', 'radio_speed_centered', 'radio_gravity_plane', 'radio_gravity_plane_centered', 'radio_frame16', 'radio_frame32', 'radio_frame_median', 'radio_frame_risk_readout', 'radio_frame_independent_cs', 'control_independent_cs']
CHECKPOINT_SHA = '5e1fc5ddf62c983523283a6eb901ca4180297c9060ffa291740cc31664636e2b'

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def prediction_job(job):
    global _CACHE
    base, checkpoint, segment, output, sensor_root, policy = job
    from support_label_barrier import deny_label_reads
    torch.set_num_threads(1)
    with deny_label_reads() as audit:
        if _CACHE is None:
            _CACHE = LOAD_PARENT(Path(base), Path(checkpoint))
        net, template, aps, _, saved = _CACHE
        config = copy.deepcopy(template)
        if policy != 'control':
            config['filter'].update(direction_policy=policy)
        parent.NativeEngine = NativeEngine if policy in ('control', 'control_balanced') else RadioDirectionEngine
        parent._ACTOR = (net, config, aps, Path(sensor_root), saved)
        position_rule = 'balanced' if policy in ('control_balanced', 'radio_balanced', 'radio_balanced_median') else None
        report = parent.predict_trajectory((base, checkpoint, segment, output, False, 1, position_rule))
        if getattr(net, 'direction_temporal_head', None):
            engine = RadioDirectionEngine.last_instance
            contexts = engine.radio_contexts
            if len(contexts) != len(engine.forecasts):
                raise ValueError('Temporal contexts do not match current scan IDs')
            np.savez_compressed(Path(output) / segment['trajectory_id'] / 'radio_temporal_context.npz', observation_ids=np.asarray([row['observation_id'] for row in engine.forecasts]), **{key: np.asarray([row[key] for row in contexts]) for key in contexts[0]})
        assert not audit['blocked_label_attempts']
    gyro_control = policy in ('control', 'control_balanced', 'radio_model_mixture', 'control_independent_cs')
    vector_control = gyro_control or policy in ('radio_frame_registration', 'radio_frame_registration_aligned', 'radio_frame_bias_cv', 'radio_spatial_basin', 'radio_balanced', 'radio_balanced_median', 'radio_gravity_plane', 'radio_gravity_plane_centered', 'radio_frame16', 'radio_frame32', 'radio_frame_median', 'radio_frame_risk_readout', 'radio_frame_independent_cs')
    report.update(direction_policy=policy, neural_weights_changed=bool(saved.get('neural_weights_changed', False)), direct_gyro_acceleration_control=policy in ('control', 'control_balanced', 'radio_model_mixture'), direct_gyro_bearing_control=gyro_control, acceleration_vector_control=vector_control, phone_heading_used_as_walking_direction=False)
    write(Path(output) / segment['trajectory_id'] / 'direction_complete.json', report)
    return report

def score_cohort(base, checkpoint, segments, output, sensor_root):

    def loader(b, c):
        net, cfg, aps, _, saved = LOAD_PARENT(b, c)
        return (net, cfg, aps, Path(sensor_root), saved)
    parent.load_actor = loader
    try:
        result = parent.score(base, checkpoint, segments, output)
    finally:
        parent.load_actor = LOAD_PARENT
    rows = json.loads((output / 'scored_predictions.json').read_text())
    result['pooled_metrics'] = parent.metrics([r['error_m'] for r in rows])
    counts = {}
    labels = {}
    for segment in segments:
        name = segment['device']
        if name not in labels:
            labels[name] = json.loads((Path(sensor_root) / name / 'evaluation_labels.json').read_text())
        with np.load(output / segment['trajectory_id'] / 'PRE_forecasts.npz', allow_pickle=False) as z:
            selected = [i for i, oid in enumerate(z['observation_ids']) if i >= 5 and str(oid) in labels[name]]
            counts[name] = counts.get(name, 0) + int(z['mask'][selected].sum())
    result['RSSI_AP_sample_counts'] = counts
    result['pooled_RSSI_MAE_db'] = sum((result['RSSI_MAE_db'][name] * count for name, count in counts.items())) / sum(counts.values())
    result['primary_reporting'] = 'All original scored points and valid AP observations pooled'
    write(output / 'complete.json', result)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--segments', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--policies', nargs='+', choices=POLICIES, default=['control', 'radio_frame_independent_cs'])
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    began = time.monotonic()
    parent_checkpoint = args.base / 'runs/counterfactual_v79_set/initial_window_evidence_phase/epoch_291.pt'
    if hashlib.sha256(parent_checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA:
        raise ValueError('Different selected checkpoint')
    checkpoint = args.checkpoint or parent_checkpoint
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if (args.output / 'prediction_complete.json').exists():
        raise FileExistsError('Preserve this completed experiment; use a new output')
    cohorts = json.loads(args.segments.read_text())
    snapshot = args.output / 'direction_source'
    snapshot.mkdir(parents=True, exist_ok=True)
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(str(source), str(snapshot / source.name))
    shutil.copytree(str(Path(__file__).parent / 'base_v79'), str(snapshot / 'base_v79'), ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
    roots = dict(original_fixed20=args.base / 'data/processed_rssi_causal_short_v1', AP_movement=args.base / 'runs/ap_movement_v79_291/experiment/preprocessed')
    references = dict(original_fixed20=args.base / 'runs/counterfactual_v79_set/initial_window_evidence_phase/fixed20_291', AP_movement=args.base / 'runs/ap_movement_v79_291/experiment/native_replay')
    jobs, reused = ([], [])
    for policy in args.policies:
        for cohort, segments in cohorts.items():
            output = args.output / policy / cohort
            output.mkdir(parents=True, exist_ok=True)
            write(output / 'runtime.json', dict(direction_policy=policy, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha, new_training=bool(args.checkpoint), original_data_preprocessing_preserved=True, future_positions_used=False, source_package='base_init'))
            for segment in segments:
                marker = output / segment['trajectory_id'] / 'direction_complete.json'
                if marker.exists():
                    report = json.loads(marker.read_text())
                    if report['direction_policy'] != policy:
                        raise ValueError('Different completed direction policy')
                    reused.append(report)
                else:
                    jobs.append((str(args.base), str(checkpoint), segment, str(output), str(roots[cohort]), policy))
    write(args.output / 'process.json', dict(status='RUNNING_SENSOR_ONLY_PREDICTION', pid=os.getpid(), policies=args.policies, trajectories_per_arm={k: len(v) for k, v in cohorts.items()}, workers=args.workers, resumed_completed_trajectories=len(reused), pending_trajectories=len(jobs)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports = list(reused)
        for report in pool.map(prediction_job, jobs):
            reports.append(report)
            if len(reports) % 12 == 0:
                print(json.dumps(dict(completed=len(reports), total=len(jobs) + len(reused), seconds=time.monotonic() - began)), flush=True)
    write(args.output / 'prediction_complete.json', dict(status='ALL_ARMS_PREDICTED_BEFORE_SCORING', labels_opened=False, reports=reports, seconds=time.monotonic() - began))
    results, comparisons = ({}, {})
    for policy in args.policies:
        results[policy], comparisons[policy] = ({}, {})
        for cohort, segments in cohorts.items():
            output = args.output / policy / cohort
            result = score_cohort(args.base, checkpoint, segments, output, roots[cohort])
            results[policy][cohort] = dict(pooled_metrics=result['pooled_metrics'], pooled_RSSI_MAE_db=result['pooled_RSSI_MAE_db'], RSSI_AP_sample_counts=result['RSSI_AP_sample_counts'])
            old = json.loads((references[cohort] / 'scored_predictions.json').read_text())
            new = json.loads((output / 'scored_predictions.json').read_text())
            old_keys = {(r['trajectory_id'], r['observation_id']) for r in old}
            new_keys = {(r['trajectory_id'], r['observation_id']) for r in new}
            assert new_keys == old_keys
            max_difference = 0.0
            for segment in segments:
                tid = segment['trajectory_id']
                prior = json.loads((references[cohort] / tid / 'predictions.json').read_text())
                current = json.loads((output / tid / 'predictions.json').read_text())
                assert [r['observation_id'] for r in prior] == [r['observation_id'] for r in current]
                delta = np.linalg.norm(np.asarray([r['xy_m'] for r in current]) - np.asarray([r['xy_m'] for r in prior]), axis=1)
                max_difference = max(max_difference, float(delta.max()))
            comparisons[policy][cohort] = dict(same_scored_IDs=True, maximum_prediction_difference_m=max_difference)
            if policy == 'control' and max_difference > 1e-06:
                raise ValueError('Parent runtime did not reproduce existing predictions')
            print(json.dumps(dict(policy=policy, cohort=cohort, **results[policy][cohort])), flush=True)
    write(args.output / 'complete.json', dict(status='COMPLETE_NATIVE_DIRECTION_COMPARISON', results=results, comparisons=comparisons, workers=args.workers, seconds=time.monotonic() - began, new_training=bool(args.checkpoint), later_true_positions_used_for_prediction=False, no_new_scoring_exclusions=True))
if __name__ == '__main__':
    main()
