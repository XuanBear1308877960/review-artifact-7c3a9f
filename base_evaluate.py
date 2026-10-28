import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from base_model import ThreeModuleNetwork
from base_native import NativeEngine
from base_train import write
_ACTOR = None
_SESSIONS = {}
AVAILABLE_CPUS = set(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None

def load_actor(base, checkpoint):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    root = Path(saved['data_root'])
    with np.load(root / 'source_map.npz', allow_pickle=False) as z:
        cache = {key: z[key] for key in z.files}
    cfg = json.loads((base / 'runs/fusion_network_v46_batched_wifi_candidates/configs/runtime_v46.json').read_text())
    net = ThreeModuleNetwork(cache, cfg['pcrt']['source_only_propagation_geometry'], saved['architecture']['variant'], saved['architecture'].get('localization_context', False), saved['architecture'].get('localization_kind') == 'precision', saved['architecture'].get('radio_likelihood') == 'gaussian_student_t_mixture', saved['architecture'].get('initial_context_scans', 1), saved['architecture'].get('motion_history_policy') == 'wifi_only_transient_cs_v2', saved['architecture'].get('calibration_geometry') == 'pre_pose_full_AP_covariance', saved['architecture'].get('innovation_memory') == 'conditional_nuisance', saved['architecture'].get('orientation_basis') == 'relative_AP_harmonics', saved['architecture'].get('mode_evidence', 'local'), saved['architecture'].get('initial_posterior', 'none'), saved['architecture'].get('ap_basis_prior', False), saved['architecture'].get('fixed_ap_basis', False), saved['architecture'].get('robust_transfer', False), saved['architecture'].get('calibrated_geometry', False), saved['architecture'].get('localization_geometry', 'none') == 'whitened_tangent_and_orthogonal', saved['architecture'].get('source_field', 'idw'), saved['architecture'].get('source_inducing', 32), saved['architecture'].get('source_lengthscale_m', 1.2), saved['architecture'].get('online_device_policy', 'cavity'), saved['architecture'].get('memory_recenter', False), saved['architecture'].get('trajectory_mode_weight', 0.0), saved['architecture'].get('cs_temporal_noise'), saved['architecture'].get('mode_spatial_evidence', 'none') == 'head_partition_ratio', saved['architecture'].get('position_integration', 'legacy'), saved['architecture'].get('region_reliability', 'none'), saved['architecture'].get('temporal_contrast', 0), saved['architecture'].get('initial_recognition', 'first'), saved['architecture'].get('field_tangent_weight', 0.0))
    net.load_state_dict(saved['state_dict'], strict=True)
    net.eval()
    from base_cs_temporal import configure_filter
    cfg['filter'] = configure_filter(cfg['filter'], net.cs_temporal_noise, cfg['anchors'])
    aps = json.loads((root / 'source_map.json').read_text())['ap_ids']
    return (net, cfg, aps, root, saved)

def predict_trajectory(arguments):
    global _ACTOR
    base, checkpoint, segment, output, map_refine, cs_iterations, position_rule = arguments
    torch.set_num_threads(1)
    if AVAILABLE_CPUS:
        os.sched_setaffinity(0, AVAILABLE_CPUS)
    began = time.monotonic()
    from support_label_barrier import deny_label_reads
    with deny_label_reads() as audit:
        if _ACTOR is None:
            _ACTOR = load_actor(Path(base), Path(checkpoint))
        net, cfg, aps, data, saved = _ACTOR
        position_rule = position_rule or saved.get('position_rule_training', 'mixed')
        name = segment['device']
        if name not in _SESSIONS:
            _SESSIONS[name] = [json.loads(line) for line in (data / name / 'online.ndjson').read_text().splitlines()]
        events = [e for e in _SESSIONS[name] if segment['start_ns'] <= e['monotonic_timestamp_ns'] < segment['end_ns']]
        engine = NativeEngine(net, aps, cfg['anchors'], segment['initial_position_xy_m'], segment['start_ns'], cfg['filter'], map_refine, cs_iterations, position_rule)
        count = 0
        for event in events:
            result = engine.consume(event)
            if result is not None:
                count += 1
                if not np.isfinite(result['xy_m']).all():
                    raise FloatingPointError('Nonfinite native position')
        folder = Path(output) / segment['trajectory_id']
        folder.mkdir(parents=True, exist_ok=True)
        write(folder / 'predictions.json', engine.outputs)
        write(folder / 'CS_diagnostics.json', engine.cs_diagnostics)
        write(folder / 'WiFi_quality_diagnostics.json', engine.radio_diagnostics)
        if map_refine:
            write(folder / 'WiFi_solver_diagnostics.json', engine.solver_diagnostics)
        rows = engine.forecasts
        np.savez_compressed(folder / 'PRE_forecasts.npz', observation_ids=np.asarray([v['observation_id'] for v in rows]), **{key: np.asarray([v[key] for v in rows]) for key in ['device_mean', 'device_covariance', 'short_residual', 'motion', 'weights', 'rssi', 'mask']}, **{'causal_innovation': np.asarray([v['causal_innovation'] for v in rows])} if net.full_pose_covariance or net.project_innovations else {}, **{'causal_device_design': np.asarray([v['causal_device_design'] for v in rows])} if net.memory_recenter else {}, **{'causal_motion_path': np.asarray([v['causal_motion_path'] for v in rows])} if net.temporal_contrast else {}, **{'headings': np.asarray([v['headings'] for v in rows])} if net.orientation_basis else {}, **{key: np.asarray([v[key] for v in rows]) for key in ['transfer_mean', 'transfer_covariance', 'transfer_offset', 'transfer_log_evidence']} if net.mode_evidence != 'local' else {}, **{'mode_spatial_log_ratio': np.asarray([v['mode_spatial_log_ratio'] for v in rows])} if net.mode_spatial_evidence else {}, **{key: np.asarray([v[key] for v in rows]) for key in ['prefix_region_information', 'prefix_region_vector']} if net.initial_posterior != 'none' else {}, **{key: np.asarray([v[key] for v in rows]) for key in ['prefix_closed_information', 'prefix_closed_vector', 'prefix_closed_probability', 'prefix_retired_information', 'prefix_retired_vector']} if net.region_reliability != 'none' else {})
        capture = folder / 'configured'
        capture.mkdir(exist_ok=True)
        np.savez_compressed(capture / 'rolling_pcrt_capture.npz', observation_ids=np.asarray([v['observation_id'] for v in rows]), times_ns=np.asarray([v['timestamp_ns'] for v in rows], dtype=np.int64), rssi=np.asarray([v['rssi'] for v in rows]), mask=np.asarray([v['mask'] for v in rows]), motion_tokens=np.asarray([v['motion_token'] for v in rows]), **{'motion_history': np.asarray([v['motion_history'] for v in rows]), 'cs_age': np.asarray([v['cs_age'] for v in rows])} if net.aligned_motion else {})
        if net.aligned_motion and engine.cs_training_rows:
            cs = engine.cs_training_rows
            np.savez_compressed(folder / 'CS_training_contexts.npz', observation_ids=np.asarray([v['observation_id'] for v in cs]), **{key: np.asarray([v[key] for v in cs]) for key in ['timestamp_ns', 'features', 'motion_history', 'range_m', 'bias_mean', 'bias_variance', 'mode_weights', 'velocity', 'anchor_index', 'wifi_index', 'wifi_age_s']}, **{key: np.asarray([v[key] for v in cs]) for key in ['temporal_mean', 'temporal_variance', 'bias_temporal_covariance']} if engine.temporal else {})
        np.savez_compressed(folder / 'native_reference_full25.npz', observation_ids=np.asarray([v['observation_id'] for v in rows]), means=np.asarray([v['pre_mean'] for v in rows]), covariances=np.asarray([v['pre_covariance'] for v in rows]), weights=np.asarray([v['weights'] for v in rows]))
        assert not audit['blocked_label_attempts']
    report = dict(status='COMPLETE_SENSOR_ONLY_PREDICTION', trajectory_id=segment['trajectory_id'], device=segment['device'], observations=count, raw_factors=len(engine.ledger), labels_opened=False, known_initial_position_only=True, seconds=time.monotonic() - began, distinct_CS=len(engine.cs_diagnostics), later_GT=False, joint_state_dimension=engine.dim, native_algorithm='V46_physical9_device6_prior_cubature_v2' + ('_MAP64' if map_refine else ''), CS_iteration_budget=cs_iterations, position_rule=position_rule, radio_likelihood=saved['architecture'].get('radio_likelihood', 'gaussian'), initial_context_scans=net.initial_context_scans, initial_support_ids=engine.initial_ids, motion_history_policy=saved['architecture'].get('motion_history_policy', 'legacy_mixed_callbacks'))
    report.update(calibration_geometry=saved['architecture'].get('calibration_geometry', 'legacy_post_pose_diagonal'), innovation_memory=saved['architecture'].get('innovation_memory', 'raw_native'), orientation_basis=saved['architecture'].get('orientation_basis', 'none'), mode_evidence=saved['architecture'].get('mode_evidence', 'local'), initial_posterior=saved['architecture'].get('initial_posterior', 'none'))
    report['cs_temporal_noise'] = net.cs_temporal_noise
    report['mode_spatial_evidence'] = saved['architecture'].get('mode_spatial_evidence', 'none')
    report['position_integration'] = net.integration_grid.description()
    report['region_reliability'] = net.region_reliability
    report['temporal_contrast'] = net.temporal_contrast
    report['initial_recognition'] = net.initial_recognition
    write(folder / 'prediction_complete.json', report)
    return report

def metrics(error):
    error = np.asarray(error, float)
    return dict(n=len(error), median_m=float(np.median(error)), p90_m=float(np.quantile(error, 0.9)), p95_m=float(np.quantile(error, 0.95)), mean_m=float(error.mean()))

def score(base, checkpoint, segments, output):
    net, cfg, aps, data, saved = load_actor(base, checkpoint)
    errors = defaultdict(list)
    radio_errors = defaultdict(list)
    device_errors = defaultdict(list)
    per_trajectory = []
    labels = {}
    rows = []
    quality = defaultdict(lambda: defaultdict(list))
    torch.set_num_threads(1)
    for segment in segments:
        name = segment['device']
        tid = segment['trajectory_id']
        folder = output / tid
        if name not in labels:
            labels[name] = json.loads((data / name / 'evaluation_labels.json').read_text())
        predictions = json.loads((folder / 'predictions.json').read_text())
        scored = predictions[4:]
        trajectory_error = []
        for prediction in scored:
            label = labels[name].get(prediction['observation_id'])
            if label is None:
                continue
            e = float(np.linalg.norm(np.asarray(prediction['xy_m']) - label['gt_xy_m']))
            errors[name].append(e)
            trajectory_error.append(e)
            rows.append(dict(device=name, trajectory_id=tid, observation_id=prediction['observation_id'], predicted_xy_m=prediction['xy_m'], reference_xy_m=label['gt_xy_m'], error_m=e))
        with np.load(folder / 'PRE_forecasts.npz', allow_pickle=False) as z:
            selected = [i for i, oid in enumerate(z['observation_ids']) if i >= 5 and str(oid) in labels[name]]
            if selected:
                xy = torch.tensor([labels[name][str(z['observation_ids'][i])]['gt_xy_m'] for i in selected], dtype=torch.float32)
                context = torch.tensor(z['motion'][selected])
                mu = torch.tensor(z['device_mean'][selected])
                delta = torch.tensor(z['short_residual'][selected])
                weight = torch.tensor(z['weights'][selected])
                with torch.no_grad():
                    field = net.field(xy, context)
                    basis = field['basis'][:, None]
                    if net.orientation_basis:
                        heading = torch.tensor(z['headings'][selected])
                        expanded = {key: value[:, None].expand(len(xy), heading.shape[1], *value.shape[1:]) for key, value in field.items()}
                        basis = net.field.condition_heading(expanded, xy[:, None], heading)['basis']
                    device_prediction = field['mean'][:, None] + (basis @ mu[..., None]).squeeze(-1)
                    device_prediction = (device_prediction * weight[..., None]).sum(1).numpy()
                    predicted = field['mean'][:, None] + (basis @ mu[..., None]).squeeze(-1) + delta
                    predicted = (predicted * weight[..., None]).sum(1).numpy()
                absolute = abs(predicted - z['rssi'][selected])
                valid = z['mask'][selected]
                radio_errors[name].extend(absolute[valid].tolist())
                device_errors[name].extend(abs(device_prediction - z['rssi'][selected])[valid].tolist())
        per_trajectory.append(dict(device=name, trajectory_id=tid, **metrics(trajectory_error)))
        path = folder / 'WiFi_quality_diagnostics.json'
        if path.exists():
            for item in json.loads(path.read_text()):
                for sigma, pi, valid in zip(item['sigma_db'], item['outlier_probability'], item['available_AP']):
                    if valid:
                        quality[name]['radio_sigma_db'].append(sigma)
                        quality[name]['radio_outlier_probability'].append(pi)
                if item['spatial_precision_gain'] is not None:
                    quality[name]['spatial_precision_gain'].extend(np.asarray(item['spatial_precision_gain']).ravel().tolist())
                if 'nominal_responsibility' in item:
                    quality[name]['radio_nominal_responsibility'].extend(item['nominal_responsibility'])
        path = folder / 'CS_diagnostics.json'
        if path.exists():
            for item in json.loads(path.read_text()):
                quality[name]['CS_sigma_m'].append(item['sigma'])
                quality[name]['CS_outlier_probability'].append(item['outlier_probability'])
    result = dict(status='COMPLETE_NATIVE_REPLAY_AND_SEPARATE_SCORING', checkpoint=str(checkpoint), checkpoint_epoch=saved['epoch'], variant=saved['architecture']['variant'], metrics={name: metrics(error) for name, error in errors.items()}, RSSI_MAE_db={name: float(np.mean(error)) for name, error in radio_errors.items()}, device_only_RSSI_MAE_db={name: float(np.mean(error)) for name, error in device_errors.items()}, RSSI_scope='Same smoothed observations; actual native causal PRE predictions at offline loss-only GT XY', prediction_labels_opened=False, initial_position_known_once=True, startup_location_skip=4, startup_RSSI_skip=5, per_trajectory=per_trajectory, architecture=saved['architecture'], goal_complete=False)
    result['uncertainty'] = {name: {key: dict(mean=float(np.mean(values)), median=float(np.median(values)), p95=float(np.quantile(values, 0.95))) for key, values in columns.items() if values} for name, columns in quality.items()}
    metadata = output / 'runtime.json'
    if metadata.exists():
        result['runtime'] = json.loads(metadata.read_text())
    result['fixed20_complete'] = len(segments) == 20 and all((result['metrics'].get(name, {}).get('n') == count for name, count in [('redmik40s', 454), ('realme2', 874)]))
    result['both_phone_thresholds_met'] = result['fixed20_complete'] and all((name in result['metrics'] and result['metrics'][name]['median_m'] < 1.0 and (result['RSSI_MAE_db'][name] <= 3.0) for name in ['redmik40s', 'realme2']))
    reference_path = base / 'runs/fusion_network_v46_batched_wifi_candidates/iteration_complete.json'
    if result['fixed20_complete'] and reference_path.exists():
        reference = json.loads(reference_path.read_text())['primary_metrics']
        difference = {name: {key: result['metrics'][name][key] - reference[name][key] for key in ['median_m', 'p90_m', 'p95_m', 'mean_m']} for name in ['redmik40s', 'realme2']}
        result['V46_difference_m'] = difference
        result['all_eight_no_worse_than_V46'] = all((value <= 0 for phone in difference.values() for value in phone.values()))
        result['V46_reference'] = str(reference_path)
    write(output / 'complete.json', result)
    write(output / 'scored_predictions.json', rows)
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--partition', choices=('train', 'validation', 'test'), required=True)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--prediction-only', action='store_true')
    p.add_argument('--map-refine', action='store_true')
    p.add_argument('--cs-iterations', type=int, choices=(1, 3), default=1)
    p.add_argument('--position-rule', choices=('mixed', 'hermite', 'balanced'))
    args = p.parse_args()
    began = time.monotonic()
    cfg = json.loads((args.base / 'runs/fusion_network_v46_batched_wifi_candidates/configs/runtime_v46.json').read_text())
    from support_protocol import selected_segments
    manifest = json.loads((Path(cfg['trajectories']['manifest']).parent / (args.partition + '.json')).read_text())
    segments = selected_segments(manifest, args.partition, cfg)
    if args.limit:
        segments = segments[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / 'runtime_source'
    snapshot.mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(str(source), str(snapshot / source.name))
    write(args.output / 'runtime.json', dict(command=sys.argv, python=sys.version, torch=torch.__version__, map_refine=args.map_refine, CS_iteration_budget=args.cs_iterations, position_rule=args.position_rule or 'checkpoint', partition=args.partition, checkpoint=str(args.checkpoint), known_start_only=True, source_snapshot=str(snapshot)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(predict_trajectory, [(str(args.base), str(args.checkpoint), s, str(args.output), args.map_refine, args.cs_iterations, args.position_rule) for s in segments]))
    write(args.output / 'prediction_complete.json', dict(status='COMPLETE', partition=args.partition, trajectories=reports, label_files_opened=False, workers=args.workers, seconds=time.monotonic() - began))
    if not args.prediction_only:
        report = score(args.base, args.checkpoint, segments, args.output)
        print(json.dumps({k: v for k, v in report.items() if k != 'per_trajectory'}), flush=True)
if __name__ == '__main__':
    main()
