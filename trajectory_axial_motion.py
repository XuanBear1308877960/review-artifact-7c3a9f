import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import hashlib
import math
import os
import pickle
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize, minimize_scalar, OptimizeResult
from scipy.special import expit, i0e, logsumexp
import torch
from base_evaluate import AVAILABLE_CPUS
from engine_utils import write
from trajectory_correlated_cs import CorrelatedPath

def axis_log_density(theta, parameters, derivative=False):
    mu = np.asarray(parameters[:2])
    kappa = np.exp(parameters[2:4])
    weight = expit(parameters[4])
    logs = np.log([weight, 1 - weight]) + kappa * np.cos(2 * (theta - mu)) - math.log(math.pi) - np.log(i0e(kappa)) - kappa
    total = logsumexp(logs)
    if not derivative:
        return total
    return (total, float(np.exp(logs - total) @ (2 * kappa * np.sin(2 * (theta - mu)))))

def feature_speeds(root):
    manifest = json.loads((root / 'feature_manifest.json').read_text())
    with np.load(root / 'features.npz', allow_pickle=False) as z:
        lengths = z['lengths'].copy()
        sensor = z['sensor'].copy()
        times = z['times'].copy()
        ids = z['ids'].copy()
    bounds = np.r_[0, np.cumsum(lengths)]
    return {(e['cohort'], e['segment']['trajectory_id']): dict(speed=sensor[bounds[i]:bounds[i + 1], 89] * 3, sigma=np.maximum(sensor[bounds[i]:bounds[i + 1], 90] * 2, 0.5), times=times[bounds[i]:bounds[i + 1]], ids=ids[bounds[i]:bounds[i + 1]], motion=sensor[bounds[i]:bounds[i + 1], 92:117]) for i, e in enumerate(manifest['entries'])}

def fit(a):
    prior = json.loads(a.prior.read_text())
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    features = feature_speeds(a.features)
    labels = {}
    angles = []
    parallel = []
    transverse = []
    stationary = []
    for ordinal, s in enumerate(segments):
        if ordinal % 5 == 0:
            continue
        name = s['device']
        if name not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target label in axial prior')
        if name not in labels:
            labels[name] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / name / 'evaluation_labels.json').read_text())
        rows = json.loads((a.capture / 'source_train' / s['trajectory_id'] / 'predictions.json').read_text())
        rows = [r for r in rows if r['observation_id'] in labels[name]]
        t = np.asarray([(r['timestamp_ns'] - s['start_ns']) / 1000000000.0 for r in rows])
        xy = np.asarray([labels[name][r['observation_id']]['gt_xy_m'] for r in rows])
        eig, vec = np.linalg.eigh(np.cov(xy.T))
        n = vec[:, -1]
        theta = math.atan2(n[1], n[0]) % math.pi
        angles.append(theta)
        normal = np.array([-n[1], n[0]])
        grid = np.arange(max(t[0], 0.5), t[-1] - 1.0 + 1e-06, 0.5)
        velocity = np.stack([np.interp(grid + 1, t, xy[:, j]) - np.interp(grid, t, xy[:, j]) for j in range(2)], -1)
        f = features['source_train', s['trajectory_id']]
        speed = np.interp(grid + 0.5, f['times'], f['speed'])
        parallel.extend((abs(velocity @ n) - speed).tolist())
        transverse.extend((velocity @ normal).tolist())
        stationary.extend((abs(velocity @ n) < 0.2).tolist())
    angles = np.asarray(angles)
    result = minimize(lambda p: -float(np.mean([axis_log_density(theta, p) for theta in angles])), [0.0, math.pi / 2, math.log(10), math.log(10), -0.5], method='L-BFGS-B', bounds=[(-math.pi, math.pi)] * 2 + [(math.log(0.1), math.log(200))] * 2 + [(-4, 4)])
    transverse = np.asarray(transverse)
    parallel = np.asarray(parallel)
    perp = minimize_scalar(lambda log_sigma: float((2 * np.log1p(transverse ** 2 / (3 * math.exp(2 * log_sigma))) + log_sigma).mean()), bounds=(math.log(0.1), math.log(2.0)), method='bounded')
    par = minimize_scalar(lambda log_sigma: float((0.5 * parallel ** 2 / math.exp(2 * log_sigma) + log_sigma).mean()), bounds=(math.log(0.3), math.log(2.0)), method='bounded')
    prior.update(axial_motion_prior_parameters=result.x.tolist(), axial_parallel_sigma_m_s=math.exp(float(par.x)), axial_transverse_sigma_m_s=math.exp(float(perp.x)), axial_stationary_probability=float(np.mean(stationary)), axial_prior_source_trajectories=43, axial_prior_target_labels_used=False, axial_prior_scope='Scene-specific source trajectory axis distribution; axis is inferred, no target route template', speed_is_learned_from_measured_IMU_norms=True)
    write(a.output, prior)
    print(json.dumps({k: v for k, v in prior.items() if k.startswith('axial_')}), flush=True)

class AxialPath(CorrelatedPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis_parameters = np.asarray(self.prior['axial_motion_prior_parameters'])
        if 'known_start_axis_logit_coefficients' in self.prior:
            from trajectory_local_axis_prior import start_features
            self.axis_parameters[4] = float(start_features(self.initial) @ np.asarray(self.prior['known_start_axis_logit_coefficients']))
        self.mid = 0.5 * (self.knots[:-1] + self.knots[1:])
        self.Vmid = self.spline(self.mid, 1)
        self.mid_speed = np.interp(self.mid, self.t, self.speed).clip(0.1, 3.0)
        self.mid_speed_evidence_sigma = np.interp(self.mid, self.t, self.speed_sigma)
        if self.prior.get('duration_motion_parameters'):
            if 'directional_velocity_parameters' in self.prior or self.prior.get('IMU_motion_context_model'):
                raise ValueError('Duration5state branch requires explicit composition with other alternative motion models')
            from trajectory_duration_motion import context as duration_context
            self.duration_motion_context = duration_context(self.prior, self.mid_speed, np.diff(self.knots), self.mid_speed_evidence_sigma)
        if self.prior.get('IMU_motion_context_model'):
            if 'directional_velocity_parameters' in self.prior:
                raise ValueError('Conditional5state context and circular family need an explicit mixture implementation')
            from trajectory_IMU_motion import context as imu_context
            self.IMU_motion_context = imu_context(self.prior, args[0].parent.name, args[0].name, self.mid, self.mid_speed, np.diff(self.knots), self.mid_speed_evidence_sigma)
        self.parallel_sigma = float(self.prior['axial_parallel_sigma_m_s'])
        self.transverse_sigma = float(self.prior['axial_transverse_sigma_m_s'])
        self.stop = float(np.clip(self.prior['axial_stationary_probability'], 0.02, 0.4))
        if self.prior.get('heading_marginal_hypotheses', 0):
            if not self.prior.get('velocity_state_temporal_selected') or self.prior.get('velocity_gain_marginalization') or self.prior.get('motion_norm_speed_evidence'):
                raise ValueError('Heading marginalization currently uses the unchanged temporal five-state motion model')
            from trajectory_heading_marginal import context as heading_context
            self.heading_context = heading_context(self.mid_speed, np.diff(self.knots), self.prior['velocity_state_parameters'], self.axis_parameters, self.mid_speed_evidence_sigma, int(self.prior['heading_marginal_hypotheses']))
        if 'directional_velocity_parameters' in self.prior:
            from trajectory_directional_velocity import context
            self.directional_velocity_context = context(self.prior['directional_velocity_parameters'], np.diff(self.knots), self.mid, self.prior.get('directional_velocity_directions', 16))
        if self.prior.get('motion_norm_speed_evidence', False):
            folder = Path(self.prior['pure_IMU_evidence_root']) / args[0].parent.name / args[0].name
            with np.load(folder / 'motion_evidence.npz', allow_pickle=False) as z:
                if z['observation_ids'].tolist() != [r['observation_id'] for r in self.predictions]:
                    raise ValueError('IMU evidence IDs differ')
                if not np.allclose(z['times_s'], self.t, atol=1e-05, rtol=0):
                    raise ValueError('IMU evidence times differ')
                self.mid_speed = np.interp(self.mid, self.t, z['speed_m_s']).clip(0.1, 3.0)
                self.mid_speed_sigma = np.interp(self.mid, self.t, z['sigma_m_s']).clip(0.5, 2.0)
                self.mid_bend = np.interp(self.mid, self.t, z['bend_probability']).clip(0, 1)

    def path_objective(self, vector, mode):
        return super().objective(vector, mode)

    def axial_objective(self, vector, mode=0):
        value, gradient = self.path_objective(vector[:-1], mode)
        q, _ = self.unpack(vector[:-1])
        theta = float(vector[-1])
        n = np.array([math.cos(theta), math.sin(theta)])
        m = np.array([-n[1], n[0]])
        velocity = self.Vmid @ q
        vp = velocity @ n
        vt = velocity @ m
        if hasattr(self, 'heading_context'):
            from trajectory_heading_marginal import energy as heading_energy
            motion, g, _ = heading_energy(velocity, self.heading_context)
            strength = self.prior.get('trajectory_factor_weights', {}).get('axial', 1.0)
            value += (motion + 0.5 * theta * theta + 0.5 * math.log(2 * math.pi)) * strength
            grad = self.Vmid.T @ g
            gradient[:2 * (self.n - 1)] += grad[1:].ravel() * strength
            return (value, np.r_[gradient, theta * strength])
        if 'velocity_state_parameters' in self.prior or 'directional_velocity_parameters' in self.prior:
            if hasattr(self, 'duration_motion_context'):
                from trajectory_IMU_motion import energy as duration_energy
                energy, g, _ = duration_energy(np.stack((vp, vt), -1), self.duration_motion_context, True)
            elif hasattr(self, 'IMU_motion_context'):
                from trajectory_IMU_motion import energy as imu_energy
                energy, g, _ = imu_energy(np.stack((vp, vt), -1), self.IMU_motion_context, True)
            elif 'directional_velocity_parameters' in self.prior:
                if self.prior.get('directional_velocity_family_marginalization', False):
                    from trajectory_motion_family import family_energy
                    energy, g, _ = family_energy(np.stack((vp, vt), -1), self.mid_speed, np.diff(self.knots), self.prior['velocity_state_parameters'], self.prior['velocity_state_temporal_selected'], self.mid_speed_evidence_sigma, self.directional_velocity_context, float(self.prior['directional_velocity_family_probability']), True)
                else:
                    from trajectory_directional_velocity import energy as directional_energy
                    energy, g, _ = directional_energy(np.stack((vp, vt), -1), self.mid_speed, self.directional_velocity_context, True)
                    fraction = float(self.prior.get('directional_velocity_fraction', 1.0))
                    if not 0.0 <= fraction <= 1.0:
                        raise ValueError('Invalid directional motion fraction')
                    if fraction < 1.0:
                        from trajectory_velocity_states import velocity_energy
                        old_energy, old_g, _ = velocity_energy(np.stack((vp, vt), -1), self.mid_speed, np.diff(self.knots), self.prior['velocity_state_parameters'], self.prior['velocity_state_temporal_selected'], True, self.mid_speed_evidence_sigma)
                        energy = fraction * energy + (1 - fraction) * old_energy
                        g = fraction * g + (1 - fraction) * old_g
            elif self.prior.get('velocity_gain_marginalization', False):
                from trajectory_motion_gain import gain_energy
                energy, g, _ = gain_energy(np.stack((vp, vt), -1), self.mid_speed, np.diff(self.knots), self.prior['velocity_state_parameters'], self.prior['velocity_state_temporal_selected'], self.prior['velocity_log_gain_prior_mean'], self.prior['velocity_log_gain_prior_sigma'], True)
            else:
                from trajectory_velocity_states import velocity_energy
                energy, g, _ = velocity_energy(np.stack((vp, vt), -1), self.mid_speed, np.diff(self.knots), self.prior['velocity_state_parameters'], self.prior['velocity_state_temporal_selected'], True, self.mid_speed_evidence_sigma)
            logaxis, daxis = axis_log_density(theta, self.axis_parameters, True)
            strength = self.prior.get('trajectory_factor_weights', {}).get('axial', 1.0)
            value += (energy - logaxis) * strength
            gv = g[:, 0, None] * n + g[:, 1, None] * m
            grad = self.Vmid.T @ gv
            gradient[:2 * (self.n - 1)] += grad[1:].ravel() * strength
            return (value, np.r_[gradient, float((g[:, 0] * vt - g[:, 1] * vp).sum() + daxis) * strength])
        if self.prior.get('motion_norm_speed_evidence', False):
            speed = np.sqrt(np.square(velocity).sum(-1) + 1e-06)
            residual = speed - self.mid_speed
            sigma = self.mid_speed_sigma
            transverse = self.prior['motion_transverse_sigma_m_s'] * np.exp(self.prior['motion_bend_log_sigma_scale'] * self.mid_bend)
            es = 2 * np.log1p(residual * residual / (3 * sigma * sigma)) + np.log(sigma)
            gs = 4 * residual / (3 * sigma * sigma + residual * residual)
            et = 2 * np.log1p(vt * vt / (3 * transverse * transverse)) + np.log(transverse)
            gt = 4 * vt / (3 * transverse * transverse + vt * vt)
            weight = np.diff(self.knots)
            logaxis, daxis = axis_log_density(theta, self.axis_parameters, True)
            strength = self.prior.get('trajectory_factor_weights', {}).get('axial', 1.0)
            value += float(((es + et) * weight).sum() - logaxis) * strength
            gv = (gs[:, None] * velocity / speed[:, None] + gt[:, None] * m) * weight[:, None]
            g = self.Vmid.T @ gv
            gradient[:2 * (self.n - 1)] += g[1:].ravel() * strength
            return (value, np.r_[gradient, float((-gt * vp * weight).sum() + daxis) * strength])
        means = np.stack((self.mid_speed, -self.mid_speed, np.zeros(len(self.mid))), -1)
        sigmas = np.array([self.parallel_sigma, self.parallel_sigma, 0.4])
        logweights = np.log([(1 - self.stop) / 2, (1 - self.stop) / 2, self.stop])
        residual = vp[:, None] - means
        logp = -0.5 * residual ** 2 / sigmas ** 2 - np.log(sigmas) - 0.5 * math.log(2 * math.pi) + logweights
        marginal = logsumexp(logp, -1)
        responsibility = np.exp(logp - marginal[:, None])
        dparallel = (responsibility * residual / sigmas ** 2).sum(-1)
        s2 = self.transverse_sigma ** 2
        lateral = 2 * np.log1p(vt * vt / (3 * s2)) + math.log(self.transverse_sigma)
        dtransverse = 4 * vt / (3 * s2 + vt * vt)
        weight = np.diff(self.knots)
        logaxis, daxis = axis_log_density(theta, self.axis_parameters, True)
        axis_weight = self.prior.get('trajectory_factor_weights', {}).get('axial', 1.0)
        value += float(((-marginal + lateral) * weight).sum() - logaxis) * axis_weight
        gv = (dparallel[:, None] * n + dtransverse[:, None] * m) * weight[:, None]
        grad = self.Vmid.T @ gv
        gradient[:2 * (self.n - 1)] += grad[1:].ravel() * axis_weight
        dtheta = float(((dparallel * vt - dtransverse * vp) * weight).sum() + daxis) * axis_weight
        return (value, np.r_[gradient, dtheta])

    def solve_axis(self, alternative, modes):
        if hasattr(self, 'polish_parent_state'):
            from trajectory_solver_polish import polish_axis
            return polish_axis(self, modes)
        outcomes = []
        initial_mixture = bool(self.prior.get('joint_marginal_initial_mixture', False))
        if hasattr(self, 'additional_parent_state'):
            parent = self.additional_parent_state['optimization_state']['vector'].copy()
            if hasattr(self, 'heading_context'):
                parent[-1] = 0.0
            value, _ = self.axial_objective(parent, 0)
            outcomes.append(OptimizeResult(x=parent, fun=value, success=False, nit=0, retained_parent=True, message='Retained sensor-only parent under current model', calibration_mode=0, axis_seed_component=0))
            result = minimize(self.axial_objective, parent, args=(0,), jac=True, method='L-BFGS-B', options=dict(maxiter=300, ftol=1e-10, gtol=2e-05, maxls=40))
            result.update(calibration_mode=0, axis_seed_component=0)
            outcomes.append(result)
        if hasattr(self, 'sensor_proposal_paths'):
            if not initial_mixture:
                raise ValueError('Sensor path proposals require the same joint initial device mixture')
            if self.prior.get('save_all_calibration_candidates', False):
                raise ValueError('Use the source joint-marginal candidate convention')
            if modes[0] == 0:
                parent = self.sensor_proposal_parent_vector.copy()
                value, _ = self.axial_objective(parent, 0)
                if abs(value - self.sensor_proposal_parent_objective) > 1e-08 * (1 + abs(value)):
                    raise ValueError('Sensor initialization changed the parent physical objective')
                outcomes.append(OptimizeResult(x=parent, fun=value, success=False, nit=0, retained_parent=True, message='Retained sensor-only parent; no new optimizer convergence claim', calibration_mode=0, axis_seed_component=int(np.argmin(abs(self.axis_parameters[:2] - parent[-1])))))
        mode_axes = [(0, axis) for axis in self.axis_parameters[:2]] if initial_mixture else [(mode, axis) for mode in modes for axis in self.axis_parameters[:2]]
        for mode, axis in mode_axes:
            native = np.asarray([r['mode_xy_m'][mode] for r in self.predictions])
            n = np.array([math.cos(axis), math.sin(axis)])
            normal = np.array([-n[1], n[0]])
            seeds = [self.learned_path, native, alternative]
            boost_paths = [self.learned_path, alternative]
            if initial_mixture:
                group = modes[0] // 2
                seeds = [np.asarray([r['mode_xy_m'][m] for r in self.predictions]) for m in modes]
                if group == 0:
                    seeds.extend([self.learned_path, alternative])
                if group == 1 and self.prior.get('joint_opposite_direction_seeds', False):
                    seeds.extend([2 * self.initial - self.learned_path, 2 * self.initial - alternative])
                if group == 1 and hasattr(self, 'counterfactual_parent_path'):
                    seeds.append(self.counterfactual_parent_path)
                if hasattr(self, 'sensor_proposal_paths'):
                    seeds.extend(self.sensor_proposal_paths[group::4])
                boost_paths = [self.learned_path] if group == 2 else [alternative] if group == 3 else []
            for path in boost_paths:
                grid = np.stack([np.interp(self.knots, self.t, path[:, j]) for j in range(2)], -1)
                derivative = np.gradient(grid, self.knots, axis=0) @ n
                derivative = gaussian_filter1d(derivative, 0.25 / np.median(np.diff(self.knots)))
                sign = np.where(derivative >= 0, 1.0, -1.0)
                speed = np.interp(self.knots, self.t, self.speed)
                longitudinal = np.r_[0.0, np.cumsum(0.5 * (sign[1:] * speed[1:] + sign[:-1] * speed[:-1]) * np.diff(self.knots))]
                lateral = (grid - grid[0]) @ normal
                boosted = self.initial + longitudinal[:, None] * n + lateral[:, None] * normal
                seeds.append(np.stack([np.interp(self.t, self.knots, boosted[:, j]) for j in range(2)], -1))
            for path in seeds:
                q = np.stack([np.interp(self.knots, self.t, path[:, j]) for j in range(2)], -1)
                q += self.initial - q[0]
                x = np.r_[q[1:].ravel(), np.zeros(4), axis]
                result = minimize(self.axial_objective, x, args=(mode,), jac=True, method='L-BFGS-B', options=dict(maxiter=int(self.prior.get('path_solver_maxiter', 300)), ftol=1e-08, gtol=0.0002, maxls=30))
                result['calibration_mode'] = int(mode)
                result['axis_seed_component'] = int(np.argmin(abs(self.axis_parameters[:2] - axis)))
                outcomes.append(result)
        result = min(outcomes, key=lambda r: r.fun)
        q, _ = self.unpack(result.x[:-1])
        xy = self.B @ q
        self.last_selected_vector = result.x.copy()
        predictions = [dict(p, xy_m=x.tolist(), inference_timing='OFFLINE_SOURCE_AXIAL_MOTION', trajectory_calibration_mode=int(result['calibration_mode']), later_position_GT_used=False) for p, x in zip(self.predictions, xy)]
        diagnostic = dict(converged=bool(result.success), iterations=int(result.nit), objective=float(result.fun), axis_degrees=float(np.rad2deg(result.x[-1]) % 180), branch_objectives=[float(r.fun) for r in outcomes], stopping_reason=str(result.message), optimization_starts=len(outcomes), branch_stopping_reasons=[str(r.message) for r in outcomes], calibration_mode=int(result['calibration_mode']), later_GT_used=False, source_scene_motion_prior=True, phone_heading_used=False)
        if initial_mixture:
            diagnostic.update(initial_device_mixture_marginalized=True, optimization_native_seed_modes=modes)
        if hasattr(self, 'heading_context'):
            from trajectory_heading_marginal import energy as heading_energy
            _, _, mass = heading_energy(self.Vmid @ q, self.heading_context)
            angles = self.heading_context['angle']
            diagnostic.update(map_axis_integrated=True, heading_hypotheses=len(mass), heading_posterior_MAP_degrees=float(np.rad2deg(angles[mass.argmax()])), heading_posterior_entropy=float(-(mass * np.log(mass.clip(1e-30))).sum()), heading_network_passes_added=0, heading_phone_yaw_used=False)
            diagnostic['axis_degrees'] = diagnostic['heading_posterior_MAP_degrees']
        if 'directional_velocity_parameters' in self.prior:
            diagnostic.update(motion_direction_states=self.prior.get('directional_velocity_directions', 16) + 1, directional_velocity_fraction=self.prior.get('directional_velocity_fraction', 1.0), phone_map_bearing_used=False)
            if self.prior.get('directional_velocity_family_marginalization', False):
                self.axial_objective(result.x, int(result['calibration_mode']))
                diagnostic.update(directional_motion_families_marginalized=True, directional_motion_family_posterior=self.directional_velocity_context['last_family_posterior'].tolist(), directional_motion_family_prior_probability=float(self.prior['directional_velocity_family_probability']), motion_models_are_alternatives_not_duplicate_measurements=True)
        if hasattr(self, 'sensor_proposal_paths'):
            diagnostic.update(sensor_grid_initializations_added=len(self.sensor_proposal_paths[modes[0] // 2::4]), sensor_grid_parent_candidate_retained=modes[0] == 0, sensor_grid_parent_objective=self.sensor_proposal_parent_objective, sensor_grid_parent_objective_recomputed_for_model=bool(self.prior.get('sensor_path_proposal_recompute_parent_objective', False)), sensor_grid_parent_candidate_selected=bool(result.get('retained_parent', False)), sensor_grid_changes_likelihood=False, sensor_grid_reference_positions_used=False)
        if self.prior.get('velocity_gain_marginalization', False):
            from trajectory_motion_gain import gain_energy, gain_components
            theta = float(result.x[-1])
            n = np.array([math.cos(theta), math.sin(theta)])
            m = np.array([-n[1], n[0]])
            velocity = self.Vmid @ q
            _, _, mass = gain_energy(np.stack((velocity @ n, velocity @ m), -1), self.mid_speed, np.diff(self.knots), self.prior['velocity_state_parameters'], self.prior['velocity_state_temporal_selected'], self.prior['velocity_log_gain_prior_mean'], self.prior['velocity_log_gain_prior_sigma'], True)
            gain = gain_components(self.prior['velocity_log_gain_prior_mean'], self.prior['velocity_log_gain_prior_sigma'])
            diagnostic.update(motion_gain_marginalized_for_whole_trajectory=True, motion_gain_components=gain.tolist(), motion_gain_component_posterior=mass.tolist(), motion_gain_posterior_mean=float(mass @ gain))
        if self.prior.get('save_all_calibration_candidates', False):
            candidates = []
            for mode in modes:
                for component in range(2):
                    best = min((r for r in outcomes if r['calibration_mode'] == mode and r['axis_seed_component'] == component), key=lambda r: r.fun)
                    knots, _ = self.unpack(best.x[:-1])
                    points = self.B @ knots
                    candidates.append(dict(calibration_mode=int(mode), candidate_id=int(mode * 2 + component), axis_seed_component=component, objective=float(best.fun), converged=bool(best.success), iterations=int(best.nit), axis_degrees=float(np.rad2deg(best.x[-1]) % 180), xy_m=points.tolist(), knots_s=self.knots.tolist(), knots_xy_m=knots.tolist()))
            diagnostic['candidate_trajectories'] = candidates
        return (predictions, diagnostic)

def job(task):
    capture, output, learned, alternative, cohort, s, prior, evidence, modes = task
    capture, output, learned, alternative = map(Path, [capture, output, learned, alternative])
    cache = None
    signature = None
    if prior.get('cache_solver_groups', False):
        signature = hashlib.sha256(json.dumps(prior, sort_keys=True).encode()).hexdigest()
        cache = output / '.solver_groups' / cohort / s['trajectory_id'] / ('group_%d.pkl' % modes[0])
        if cache.exists():
            with cache.open('rb') as handle:
                saved = pickle.load(handle)
            if saved['prior_signature'] != signature:
                raise ValueError('Solver cache belongs to another prior')
            return saved['result']
    torch.set_num_threads(1)
    if AVAILABLE_CPUS:
        os.sched_setaffinity(0, AVAILABLE_CPUS)
    source = capture / cohort / s['trajectory_id']
    target = output / cohort / s['trajectory_id']
    from support_label_barrier import deny_label_reads
    with deny_label_reads() as audit:
        rows = json.loads((learned / cohort / s['trajectory_id'] / 'predictions.json').read_text())
        other = json.loads((alternative / cohort / s['trajectory_id'] / 'predictions.json').read_text())
        if [r['observation_id'] for r in rows] != evidence['ids'].tolist():
            raise ValueError('Motion evidence IDs differ')
        path = np.asarray([r['xy_m'] for r in rows])
        speed = (evidence['speed'], evidence['sigma'], np.ones(len(path), bool))
        solver_class = AxialPath
        extra = {}
        if prior.get('receiver_angular_sigma_db', 0.0) > 0:
            from trajectory_receiver_angular import ReceiverAngularPath
            solver_class = ReceiverAngularPath
            extra['force_context'] = evidence
        elif prior.get('receiver_curvature_sigma_db', 0.0) > 0:
            from trajectory_receiver_curvature import ReceiverCurvaturePath
            solver_class = ReceiverCurvaturePath
            extra['force_context'] = evidence
        elif prior.get('RSSI_raw_history_noise_source_selected', False):
            from trajectory_RSSI_history_noise import RawHistoryRSSINoisePath
            solver_class = RawHistoryRSSINoisePath
            extra['force_context'] = evidence
        elif prior.get('RSSI_history_noise_source_selected', False):
            from trajectory_RSSI_history_noise import HistoryRSSINoisePath
            solver_class = HistoryRSSINoisePath
            extra['force_context'] = evidence
        elif prior.get('RSSI_map_variance_covariance', False):
            from trajectory_map_variance import MapVariancePath
            solver_class = MapVariancePath
            extra['force_context'] = evidence
        elif prior.get('RSSI_spatial_residual_covariance', False):
            from trajectory_spatial_residual import SpatialResidualPath
            solver_class = SpatialResidualPath
            extra['force_context'] = evidence
        elif prior.get('RSSI_cross_region_objective', False):
            from trajectory_cross_region import CrossRegionPath
            solver_class = CrossRegionPath
            extra['force_context'] = evidence
        elif prior.get('RSSI_cross_AP_noise', False):
            from trajectory_cross_ap_radio import CrossAPJointPath
            solver_class = CrossAPJointPath
            extra['force_context'] = evidence
        elif prior.get('temporal_RSSI_observation_model', False):
            from trajectory_temporal_radio import TemporalJointPath
            solver_class = TemporalJointPath
            extra['force_context'] = evidence
        elif prior.get('joint_marginal_RSSI', False):
            from trajectory_joint_marginal import JointMarginalPath
            solver_class = JointMarginalPath
            extra['force_context'] = evidence
        elif 'raw_HMM_parameters' in prior:
            from trajectory_raw_hmm import RawHMMPath, ForceRawHMMPath
            solver_class = RawHMMPath
            if 'marginal_force_parameters' in prior:
                solver_class = ForceRawHMMPath
                extra['force_context'] = evidence
        elif 'marginal_force_parameters' in prior:
            from trajectory_force_marginal import ForceDualPath
            solver_class = ForceDualPath
            extra['force_context'] = evidence
        elif 'dual_CS_parameters' in prior:
            from trajectory_dual_cs import DualAxialPath
            solver_class = DualAxialPath
        solver = solver_class(source, s, prior, speed, False, path, **extra)
        if prior.get('additional_parent_solver_root'):
            parent = Path(prior['additional_parent_solver_root']) / '.solver_groups' / cohort / s['trajectory_id'] / ('group_%d.pkl' % modes[0])
            with parent.open('rb') as handle:
                old = pickle.load(handle)['result']
            if old['diagnostic'].get('later_GT_used', False):
                raise ValueError('Warm parent used later positions')
            if [(row['observation_id'], row['timestamp_ns']) for row in old['predictions']] != [(row['observation_id'], row['timestamp_ns']) for row in solver.predictions]:
                raise ValueError('Warm parent observations differ')
            if not np.allclose(old['optimization_state']['knots_s'], solver.knots, atol=1e-10):
                raise ValueError('Warm parent spline differs')
            solver.additional_parent_state = old
        if prior.get('solver_polish_parent_root'):
            parent = Path(prior['solver_polish_parent_root']) / '.solver_groups' / cohort / s['trajectory_id'] / ('group_%d.pkl' % modes[0])
            with parent.open('rb') as handle:
                original = pickle.load(handle)
            model_prior = {k: v for k, v in prior.items() if not k.startswith('solver_polish_')}
            expected = hashlib.sha256(json.dumps(model_prior, sort_keys=True).encode()).hexdigest()
            if original['prior_signature'] != expected:
                raise ValueError('Polish changed the parent model configuration')
            solver.polish_parent_state = original['result']
        if prior.get('sensor_path_proposal_root'):
            if prior.get('solver_polish_parent_root'):
                raise ValueError('Do not bypass new starts with the same-model polishing shortcut')
            proposal = Path(prior['sensor_path_proposal_root']) / cohort / s['trajectory_id'] / 'proposals.npz'
            with np.load(proposal, allow_pickle=False) as z:
                if z['observation_ids'].tolist() != [r['observation_id'] for r in solver.predictions] or not np.array_equal(z['times_ns'], solver.times_ns):
                    raise ValueError('Sensor proposal observation IDs or timestamps differ')
                if not np.allclose(z['parent_knots_s'], solver.knots, atol=1e-10):
                    raise ValueError('Sensor proposal spline differs')
                solver.sensor_proposal_paths = z['paths_xy_m'].copy()
                solver.sensor_proposal_parent_vector = z['parent_vector'].copy()
                solver.sensor_proposal_parent_objective = float(z['parent_objective'])
                if prior.get('sensor_path_proposal_recompute_parent_objective', False):
                    solver.sensor_proposal_parent_objective = float(solver.axial_objective(solver.sensor_proposal_parent_vector, 0)[0])
        predictions, diagnostic = solver.solve_axis(np.asarray([r['xy_m'] for r in other]), modes)
        assert not audit['blocked_label_attempts']
    result = dict(cohort=cohort, trajectory_id=s['trajectory_id'], predictions=predictions, diagnostic=diagnostic)
    if hasattr(solver, 'calibration_forecasts'):
        result['calibration_forecasts'] = solver.calibration_forecasts
    if prior.get('save_solver_state', False):
        result['optimization_state'] = dict(vector=solver.last_selected_vector.copy(), knots_s=solver.knots.copy())
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix('.pending')
        with temporary.open('wb') as handle:
            pickle.dump(dict(prior_signature=signature, result=result), handle, protocol=4)
        temporary.replace(cache)
    return result

def infer(a):
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    segments = json.loads((a.capture / 'segments.json').read_text())
    features = feature_speeds(a.features)
    fold = getattr(a, 'source_fold', 'all')
    if fold != 'all' and a.cohorts != ['source_train']:
        raise ValueError('Source-fold inference cannot include target cohorts')
    selected = [(c, s) for c in a.cohorts for ordinal, s in enumerate(segments[c]) if fold == 'all' or (ordinal % 5 == 0) == (fold == 'heldout')]
    jobs = [(str(a.capture), str(a.output), str(a.learned_path), str(a.alternative), c, s, prior, features[c, s['trajectory_id']], list(range(group * 2, group * 2 + 2))) for c, s in selected for group in range(4)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        partials = []
        for r in pool.map(job, jobs):
            partials.append(r)
            if len(partials) % 16 == 0:
                print(json.dumps(dict(completed_groups=len(partials), total_groups=len(jobs), seconds=time.monotonic() - began)), flush=True)
    grouped = {}
    for r in partials:
        grouped.setdefault((r['cohort'], r['trajectory_id']), []).append(r)
    reports = []
    for (cohort, tid), rows in grouped.items():
        assert len(rows) == 4
        best = min(rows, key=lambda r: r['diagnostic']['objective'])
        target = a.output / cohort / tid
        target.mkdir(parents=True, exist_ok=True)
        write(target / 'predictions.json', best['predictions'])
        diagnostic = best['diagnostic']
        diagnostic['parallel_group_best_objectives'] = [r['diagnostic']['objective'] for r in rows]
        if prior.get('save_all_calibration_candidates', False):
            candidates = []
            for row in rows:
                candidates.extend(row['diagnostic'].pop('candidate_trajectories'))
            candidates.sort(key=lambda c: c['candidate_id'])
            if [c['candidate_id'] for c in candidates] != list(range(16)):
                raise ValueError('Missing or repeated calibration trajectory candidate')
            write(target / 'candidate_trajectories.json', dict(candidates=candidates, candidate_generation_later_GT_used=False, initial_position_only=True))
        write(target / 'axial_solver.json', diagnostic)
        for name in ['PRE_forecasts.npz', 'CS_diagnostics.json', 'WiFi_quality_diagnostics.json']:
            shutil.copyfile(str(a.capture / cohort / tid / name), str(target / name))
        if 'calibration_forecasts' in best:
            with np.load(target / 'PRE_forecasts.npz', allow_pickle=False) as z:
                pre = {k: z[k].copy() for k in z.files}
            prediction = best['calibration_forecasts']
            mode = int(prediction['mode'])
            for key in ['device_mean', 'device_covariance', 'short_residual']:
                pre[key][:, mode] = prediction[key]
            if 'field_motion' in prediction:
                pre['motion'] = prediction['field_motion']
            if 'receiver_angular_mean_db' in prediction:
                np.savez_compressed(target / 'receiver_angular_forecast.npz', observation_ids=pre['observation_ids'], times_ns=np.asarray([row['timestamp_ns'] for row in best['predictions']], np.int64), mean_db=prediction['receiver_angular_mean_db'], covariance_db2=prediction['receiver_angular_covariance_db2'], query_horizontal=prediction['receiver_angular_query_horizontal'], query_valid_weight=prediction['receiver_angular_query_valid_weight'], initial_horizontal=prediction['receiver_angular_initial_horizontal'])
            if 'receiver_curvature_mean_db' in prediction:
                np.savez_compressed(target / 'receiver_curvature_forecast.npz', observation_ids=pre['observation_ids'], times_ns=np.asarray([row['timestamp_ns'] for row in best['predictions']], np.int64), mean_db=prediction['receiver_curvature_mean_db'], variance_db2=prediction['receiver_curvature_variance_db2'], initial_mean_db=prediction['receiver_curvature_initial_mean_db'])
            pre['weights'][:] = 0.0
            pre['weights'][:, mode] = 1.0
            np.savez_compressed(target / 'PRE_forecasts.npz', **pre)
        reports.append(dict(cohort=cohort, trajectory_id=tid, **diagnostic))
    write(a.output / 'prediction_complete.json', dict(status='COMPLETE_SOURCE_AXIAL_SENSOR_ONLY_PATHS', reports=reports, source_prior=prior, later_GT=False, offline_future_sensors=True, CS_ranges_replaced=False, source_scene_mobility_prior=True, parallel_groups_per_trajectory=4, CPU_workers=a.workers, source_fold=fold, source_spatial_support_strength=prior.get('source_spatial_support_strength', 0.0), source_spatial_support_scope=prior.get('source_spatial_support_scope'), seconds=time.monotonic() - began, offline_selected_calibration_branch=bool(prior.get('joint_marginal_RSSI', False)), custom_RSSI_field='OPPO_only_per_AP_spatial_metric_field' if prior.get('source_map_anisotropic_model') else 'OPPO_anchored_source_trajectory_refitted_mean' if prior.get('joint_source_field_refit_model') else 'OPPO_only_smooth_RP_contrast_restoration' if prior.get('source_RP_contrast_model') else 'OPPO_only_physical_mean_spatial_GP' if prior.get('source_spatial_GP_model') else 'FiniteRadioField_radius_0.15m' if prior.get('joint_marginal_RSSI', False) else None))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['fit', 'infer'], required=True)
    for name in ['base', 'capture', 'features', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--learned-path', type=Path)
    p.add_argument('--alternative', type=Path)
    p.add_argument('--cohorts', nargs='+', default=['source_train'])
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--source-fold', choices=['all', 'train', 'heldout'], default='all')
    a = p.parse_args()
    torch.set_num_threads(1)
    if AVAILABLE_CPUS:
        os.sched_setaffinity(0, AVAILABLE_CPUS)
    if a.stage == 'fit':
        fit(a)
    else:
        infer(a)
if __name__ == '__main__':
    main()
