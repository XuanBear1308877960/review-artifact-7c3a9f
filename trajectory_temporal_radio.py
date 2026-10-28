import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular, cho_solve
from scipy.optimize import minimize
from engine_utils import write
from trajectory_joint_calibration import field
from trajectory_joint_marginal import initial_prior, mixture_posterior, JointMarginalPath, score
from trajectory_raw_hmm import ForceRawHMMPath

class ObservationModel:

    def __init__(self, query_file, pre):
        with np.load(query_file, allow_pickle=False) as z:
            q = {k: z[k].copy() for k in z.files}
        if not np.array_equal(q['observation_ids'], pre['observation_ids']):
            raise ValueError('RSSI contributor IDs differ')
        self.pre = pre
        self.N = len(pre['rssi'])
        self.S = []
        self.raw_dt = []
        self.context = []
        self.raw_times = []
        self.ap = []
        self.bounds = [0]
        for ap in range(4):
            times = q['raw_times_s_' + str(ap)]
            indices = q['raw_index_' + str(ap)]
            S = np.zeros((self.N, len(times)))
            ctx = np.zeros(len(times), np.int64)
            for i in range(self.N):
                for j in range(2):
                    rid = int(indices[i, j])
                    if rid >= 0:
                        S[i, rid] += q['query_weights'][i, ap, j]
                        ctx[rid] = q['query_context_index'][i, ap, j]
            self.S.append(S)
            self.raw_dt.append(abs(times[:, None] - times[None, :]))
            self.raw_times.extend(times)
            self.context.extend(ctx)
            self.ap.extend([ap] * len(times))
            self.bounds.append(self.bounds[-1] + len(times))
        self.raw_times = np.asarray(self.raw_times)
        self.context = np.asarray(self.context)
        self.ap = np.asarray(self.ap)
        self.mask = pre['mask'].copy() & ~q['initial_support_raw_reused']
        self.mask[:5] = False
        self.shared = [S @ S.T for S in self.S]
        self.query_metadata = q

    def freeze_field_context(self, index=4):
        if len(self.pre['motion']) <= index:
            raise ValueError('Incomplete initial-five field context')
        self.fixed_context = np.broadcast_to(self.pre['motion'][index], (len(self.raw_times), 32)).copy()

    def raw_context(self):
        return self.fixed_context if hasattr(self, 'fixed_context') else self.pre['motion'][self.context]

    def lock_device_basis(self, known_initial_xy, radio_field):
        points = np.broadcast_to(np.asarray(known_initial_xy, float), (len(self.raw_times), 2))
        _, self.fixed_device_basis = radio_field.ap_query(points, self.raw_context(), self.ap, False)

    def temper_device_spatial_change(self, known_initial_xy, radio_field, strength):
        if not 0 <= strength <= 1:
            raise ValueError('Device spatial-change strength must be between zero and one')
        points = np.broadcast_to(np.asarray(known_initial_xy, float), (len(self.raw_times), 2))
        _, self.initial_location_basis = radio_field.ap_query(points, self.raw_context(), self.ap, False)
        self.device_spatial_strength = float(strength)

    def geometry(self, raw_xy, radio_field, derivative=False):
        if hasattr(self, 'fixed_device_basis'):
            if derivative:
                raw_mean, raw_dm = radio_field.source_ap(raw_xy, self.ap, True)
            else:
                raw_mean = radio_field.source_ap(raw_xy, self.ap)
                raw_dm = None
            raw_basis = self.fixed_device_basis.copy()
            raw_basis[:, 1] = raw_mean / 50
            raw_db = np.zeros((len(raw_xy), 6, 2)) if derivative else None
            if derivative:
                raw_db[:, 1] = raw_dm / 50
        elif hasattr(radio_field, 'ap_query'):
            outputs = radio_field.ap_query(raw_xy, self.raw_context(), self.ap, derivative)
            raw_mean, raw_basis = outputs[:2]
            raw_dm, raw_db = outputs[2:4] if derivative else (None, None)
        else:
            outputs = radio_field(raw_xy, self.raw_context(), derivative)
            index = np.arange(len(raw_xy))
            raw_mean = outputs[0][index, self.ap]
            raw_basis = outputs[1][index, self.ap]
            raw_dm, raw_db = (outputs[2][index, self.ap], outputs[3][index, self.ap]) if derivative else (None, None)
        if hasattr(self, 'initial_location_basis'):
            alpha = self.device_spatial_strength
            raw_basis = raw_basis.copy()
            raw_basis[:, 2:] = self.initial_location_basis[:, 2:] + alpha * (raw_basis[:, 2:] - self.initial_location_basis[:, 2:])
            if derivative:
                raw_db = raw_db.copy()
                raw_db[:, 2:] *= alpha
        mean = np.zeros((self.N, 4))
        basis = np.zeros((self.N, 4, 6))
        for ap in range(4):
            sl = slice(self.bounds[ap], self.bounds[ap + 1])
            mean[:, ap] = self.S[ap] @ raw_mean[sl]
            basis[:, ap] = self.S[ap] @ raw_basis[sl]
        if not derivative:
            return (mean, basis)
        return (mean, basis, raw_dm, raw_db)

    def noise(self, parameters):
        tau = math.exp(parameters[8])
        cov = []
        shared = []
        derivatives = []
        for ap in range(4):
            S = self.S[ap]
            K = np.exp(-self.raw_dt[ap] / tau)
            prop = S @ K @ S.T
            d_tau = S @ (K * self.raw_dt[ap] / tau) @ S.T
            white = math.exp(2 * parameters[ap])
            colored = math.exp(2 * parameters[4 + ap])
            raw = math.exp(2 * parameters[9 + ap])
            correlated = raw * self.shared[ap] + colored * prop
            cov.append(correlated + np.eye(self.N) * (white + 0.1 ** 2))
            shared.append(correlated)
            derivatives.append((2 * white * np.eye(self.N), 2 * colored * prop, colored * d_tau, 2 * raw * self.shared[ap]))
        return (cov, shared, derivatives)

    def blocks(self, covariances, mask=None):
        available = self.mask if mask is None else self.mask & mask
        blocks = []
        for ap in range(4):
            ix = np.where(available[:, ap])[0]
            if len(ix):
                blocks.append((ap, ix, cholesky(covariances[ap][np.ix_(ix, ix)], lower=True, check_finite=False)))
        return blocks

    def whiten(self, mean, basis, blocks):
        zs = []
        As = []
        ld = 0.0
        for ap, ix, L in blocks:
            zs.append(solve_triangular(L, self.pre['rssi'][ix, ap] - mean[ix, ap], lower=True, check_finite=False))
            As.append(solve_triangular(L, basis[ix, ap], lower=True, check_finite=False))
            ld += 2 * np.log(np.diag(L)).sum()
        return (np.concatenate(zs), np.concatenate(As), float(ld)) if zs else (np.zeros(0), np.zeros((0, 6)), 0.0)

    def pullback(self, g_mean, g_basis, raw_dm, raw_db, raw_spline):
        gm = []
        gb = []
        for ap in range(4):
            gm.append(self.S[ap].T @ g_mean[:, ap])
            gb.append(self.S[ap].T @ g_basis[:, ap])
        gm = np.concatenate(gm)
        gb = np.concatenate(gb)
        points = gm[:, None] * raw_dm + np.einsum('nd,ndz->nz', gb, raw_db)
        return raw_spline.T @ points

def fit(a):
    began = time.monotonic()
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    labels = {}
    rows = []
    prior = json.loads(a.prior.read_text())
    radio_field = field(a.base, prior)
    if getattr(a, 'freeze_field_context', False):
        prior['radio_field_context_mode'] = 'initial_five_final'
    if getattr(a, 'lock_spatial_basis', False):
        prior['geometry_locked_device_basis'] = True
    for ordinal, s in enumerate(segments):
        name = s['device']
        if name not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target label entered temporal radio noise fit')
        if name not in labels:
            labels[name] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / name / 'evaluation_labels.json').read_text())
        folder = a.capture / 'source_train' / s['trajectory_id']
        with np.load(folder / 'PRE_forecasts.npz', allow_pickle=False) as z:
            pre = {k: z[k].copy() for k in z.files}
        pred = json.loads((folder / 'predictions.json').read_text())
        selected = [p for p in pred if p['observation_id'] in labels[name]]
        t = np.r_[0.0, [(p['timestamp_ns'] - s['start_ns']) / 1000000000.0 for p in selected]]
        xy = np.vstack((s['initial_position_xy_m'], [labels[name][p['observation_id']]['gt_xy_m'] for p in selected]))
        model = ObservationModel(a.queries / 'source_train' / s['trajectory_id'] / 'radio_queries.npz', pre)
        if prior.get('radio_field_context_mode') == 'initial_five_final':
            model.freeze_field_context()
        if prior.get('geometry_locked_device_basis', False):
            model.lock_device_basis(s['initial_position_xy_m'], radio_field)
        if prior.get('device_spatial_basis_strength', 1.0) < 1:
            model.temper_device_spatial_change(s['initial_position_xy_m'], radio_field, prior['device_spatial_basis_strength'])
        model.mask &= np.asarray([str(oid) in labels[name] for oid in pre['observation_ids']])[:, None]
        raw_xy = np.stack([np.interp(model.raw_times.clip(0, t[-1]), t, xy[:, j]) for j in range(2)], -1)
        mean, basis = model.geometry(raw_xy, radio_field)
        rows.append(dict(model=model, mean=mean, basis=basis, prior=initial_prior(pre), heldout=ordinal % 5 == 0))

    def objective(parameters, heldout, derivative=True):
        total = 0.0
        count = 0
        gradient = np.zeros(13)
        for row in rows:
            if row['heldout'] != heldout:
                continue
            model = row['model']
            cov, _, dC = model.noise(parameters)
            blocks = model.blocks(cov)
            z, A, ld = model.whiten(row['mean'], row['basis'], blocks)
            loss, eta, P, _, _ = mixture_posterior(z, A, row['prior'], ld)
            total += loss
            count += len(z)
            if derivative:
                for ap, ix, L in blocks:
                    raw_residual = model.pre['rssi'][ix, ap] - row['mean'][ix, ap] - row['basis'][ix, ap] @ eta
                    precision = cho_solve((L, True), np.eye(len(ix)), check_finite=False)
                    score = precision @ raw_residual
                    W = precision @ row['basis'][ix, ap]
                    G = 0.5 * (precision - np.outer(score, score) - W @ P @ W.T)
                    for parameter, dc in zip([ap, 4 + ap, 8, 9 + ap], dC[ap]):
                        gradient[parameter] += float((G * dc[np.ix_(ix, ix)]).sum())
        return (total / count, gradient / count) if derivative else float(total / count)
    initial = np.log([1.5] * 4 + [2.0] * 4 + [1.0] + [1.0] * 4)
    direction = np.random.RandomState(81943).normal(size=13)
    direction /= np.linalg.norm(direction)
    h = 0.0001
    _, g = objective(initial, False)
    numeric = (objective(initial + h * direction, False, False) - objective(initial - h * direction, False, False)) / (2 * h)
    if abs(numeric - g @ direction) > 3e-05 * (1 + abs(numeric)):
        raise ValueError('Temporal radio source covariance gradient mismatch')
    bounds = [(math.log(0.2), math.log(12.0))] * 4 + [(math.log(0.02), math.log(15.0))] * 4 + [(math.log(0.1), math.log(30.0))] + [(math.log(0.02), math.log(12.0))] * 4
    result = minimize(objective, initial, args=(False, True), jac=True, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=100, ftol=2e-07))
    prior.update(temporal_RSSI_observation_model=True, joint_marginal_RSSI=True, joint_marginal_initial_mixture=True, save_all_calibration_candidates=False, temporal_RSSI_parameters=result.x.tolist(), temporal_RSSI_query_root=str(a.queries.resolve()), joint_runtime_base=str(a.base.resolve()), temporal_RSSI_source_train_NLL=float(result.fun), temporal_RSSI_source_heldout_NLL=objective(result.x, True, False), temporal_RSSI_fit_converged=bool(result.success), temporal_RSSI_fit_termination=str(result.message), temporal_RSSI_covariance_source_gradient_difference=float(abs(numeric - g @ direction)), rolling_RSSI_sharing_initial_raw_samples_recounted=False, measured_RSSI_values_changed=False, temporal_RSSI_target_GT_used=False, temporal_RSSI_fit_seconds=time.monotonic() - began)
    write(a.output, prior)
    print(json.dumps(dict(source_heldout_NLL=prior['temporal_RSSI_source_heldout_NLL'], white_sigma_db=np.exp(result.x[:4]).tolist(), colored_sigma_db=np.exp(result.x[4:8]).tolist(), raw_sample_sigma_db=np.exp(result.x[9:13]).tolist(), propagation_tau_s=float(np.exp(result.x[8])), seconds=time.monotonic() - began)), flush=True)

class TemporalJointPath(JointMarginalPath):

    def __init__(self, *args, **kwargs):
        ForceRawHMMPath.__init__(self, *args, **kwargs)
        self.wifi_weight[:] = 0.0
        with np.load(args[0] / 'PRE_forecasts.npz', allow_pickle=False) as z:
            self.pre = {k: z[k].copy() for k in z.files}
        if self.prior.get('field_context_override_root'):
            with np.load(Path(self.prior['field_context_override_root']) / args[0].parent.name / args[0].name / 'field_context.npz', allow_pickle=False) as z:
                if not np.array_equal(z['observation_ids'], self.pre['observation_ids']) or not np.array_equal(z['times_ns'], self.times_ns):
                    raise ValueError('Rolling field context override IDs or times differ')
                self.pre['motion'] = z['motion'].copy()
        from trajectory_robust_initial import robust_initial_prior
        self.device_prior = robust_initial_prior(self.pre, self.prior)
        self.radio_field = field(Path(self.prior['joint_runtime_base']), self.prior)
        if self.prior.get('RSSI_AP_scale_marginalization', False) and any((self.prior.get(k) for k in ['known_start_radio_conditioning', 'initial_position_device_dependency_strength', 'RSSI_segment_scale_marginalization'])):
            raise ValueError('Evaluate AP noise families separately from alternative initial/noise conditioning')
        self.observations = ObservationModel(Path(self.prior['temporal_RSSI_query_root']) / args[0].parent.name / args[0].name / 'radio_queries.npz', self.pre)
        if self.prior.get('radio_field_context_mode') == 'initial_five_final':
            self.observations.freeze_field_context()
        if self.prior.get('geometry_locked_device_basis', False):
            self.observations.lock_device_basis(self.s['initial_position_xy_m'], self.radio_field)
        if self.prior.get('device_spatial_basis_strength', 1.0) < 1:
            self.observations.temper_device_spatial_change(self.s['initial_position_xy_m'], self.radio_field, self.prior['device_spatial_basis_strength'])
        self.raw_spline = self.spline(self.observations.raw_times.clip(0, self.knots[-1]))
        self.radio_covariances, self.radio_shared_covariances, _ = self.observations.noise(np.asarray(self.prior['temporal_RSSI_parameters']))
        if self.prior.get('known_start_radio_conditioning', False):
            from trajectory_known_start_radio import configure
            self.observations, self.device_prior, self.radio_covariances, self.radio_shared_covariances, self.known_start_diagnostic = configure(self.pre, self.prior, self.s['initial_position_xy_m'], self.radio_field, self.observations, self.radio_covariances, self.radio_shared_covariances)
        self.radio_blocks = self.observations.blocks(self.radio_covariances)
        if self.prior.get('initial_position_device_dependency_strength', 0.0) > 0:
            if self.prior.get('known_start_radio_conditioning') or self.prior.get('RSSI_segment_scale_marginalization') or self.prior.get('initial_population_mixture_weight', 0.0) > 0:
                raise ValueError('Evaluate the initial-dependency mechanism before combining alternative initial/scale priors')
            from trajectory_initial_dependency import InitialDependency
            self.initial_dependency = InitialDependency(args[0], self.pre, self.prior)
            self.initial_dependency_basis = self.spline(self.t[5])
        if self.prior.get('frozen_M3_spatial_head_scale', 0.0) > 0:
            with np.load(Path(self.prior['frozen_M3_spatial_head_root']) / args[0].parent.name / args[0].name / 'head.npz', allow_pickle=False) as z:
                if not np.array_equal(z['observation_ids'], self.pre['observation_ids']) or not np.array_equal(z['times_ns'], self.times_ns):
                    raise ValueError('Frozen M3 head observation IDs or times differ')
                self.M3_head_grid = z['grid_xy'].copy()
                self.M3_head_potential = z['potential'].copy()
                self.M3_head_weight = z['weight'].copy()

    def path_objective(self, vector, mode):
        value, gradient = super().path_objective(vector, mode)
        strength = self.prior.get('source_spatial_support_strength', 0.0)
        if strength > 0:
            from trajectory_spatial_support import density
            q, _ = self.unpack(vector)
            B = self.spline(self.mid)
            energy, g = density(B @ q, np.asarray(self.prior['source_spatial_support_centres_m']), self.prior['source_spatial_support_sigma_m'], True)
            weight = np.diff(self.knots) * strength
            value += float(energy @ weight)
            dq = B.T @ (g * weight[:, None])
            gradient[:2 * (self.n - 1)] += dq[1:].ravel()
        strength = self.prior.get('frozen_M3_spatial_head_scale', 0.0)
        if strength > 0:
            from trajectory_frozen_spatial_head import potential_energy
            q, _ = self.unpack(vector)
            energy, g = potential_energy(self.B @ q, self.M3_head_grid, self.M3_head_potential, self.M3_head_weight, self.prior['frozen_M3_spatial_head_bandwidth_m'], True)
            value += strength * energy
            dq = self.B.T @ g
            gradient[:2 * (self.n - 1)] += strength * dq[1:].ravel()
        return (value, gradient)

    def radio(self, q, mode):
        model = self.observations
        mean, basis, dm, db = model.geometry(self.raw_spline @ q, self.radio_field, True)
        z, A, ld = model.whiten(mean, basis, self.radio_blocks)
        if hasattr(self, 'initial_dependency'):
            self.device_prior = self.initial_dependency.at(self.initial_dependency_basis @ q)
        if self.prior.get('RSSI_segment_scale_marginalization', False) or self.prior.get('RSSI_AP_scale_marginalization', False):
            if self.prior.get('RSSI_segment_scale_marginalization', False) and self.prior.get('RSSI_AP_scale_marginalization', False):
                raise ValueError('Do not count both global and per-AP noise mixtures')
            if self.prior.get('RSSI_AP_scale_marginalization', False):
                from trajectory_AP_noise_mixture import ap_posterior
                loss, eta, P, residual, components, derivatives = ap_posterior(z, A, self.device_prior, ld, self.radio_blocks, self.prior, True)
            else:
                from trajectory_radio_scale_mixture import scale_posterior
                loss, eta, P, residual, components, derivatives = scale_posterior(z, A, self.device_prior, ld, self.prior, True)
            gm = np.zeros_like(mean)
            gb = np.zeros_like(basis)
            offset = 0
            for ap, ix, L in self.radio_blocks:
                end = offset + len(ix)
                gm[ix, ap] = -solve_triangular(L.T, derivatives['g_z'][offset:end], lower=False, check_finite=False)
                gb[ix, ap] = solve_triangular(L.T, derivatives['g_A'][offset:end], lower=False, check_finite=False)
                offset = end
            return (loss, model.pullback(gm, gb, dm, db, self.raw_spline), (eta, P, components))
        if hasattr(self, 'initial_dependency'):
            loss, eta, P, residual, components, g_initial = self.initial_dependency.posterior(z, A, self.device_prior, ld, True)
        else:
            loss, eta, P, residual, components = mixture_posterior(z, A, self.device_prior, ld)
        gm = np.zeros_like(mean)
        gb = np.zeros_like(basis)
        offset = 0
        for ap, ix, L in self.radio_blocks:
            end = offset + len(ix)
            score = solve_triangular(L.T, residual[offset:end], lower=False, check_finite=False)
            uncertainty = solve_triangular(L.T, A[offset:end] @ P, lower=False, check_finite=False)
            gm[ix, ap] = -score
            gb[ix, ap] = uncertainty - score[:, None] * eta[None]
            offset = end
        gradient = model.pullback(gm, gb, dm, db, self.raw_spline)
        if hasattr(self, 'initial_dependency'):
            gradient += self.initial_dependency_basis[:, None] * g_initial[None]
        return (float(loss), gradient, (eta, P, components))

    def device_posterior(self, z, A, ld=0.0, blocks=None):
        if self.prior.get('RSSI_AP_scale_marginalization', False):
            from trajectory_AP_noise_mixture import ap_posterior
            if blocks is None and len(z):
                raise ValueError('Per-AP posterior requires the actual conditioning block lengths')
            return ap_posterior(z, A, self.device_prior, ld, blocks or [], self.prior)
        if self.prior.get('RSSI_segment_scale_marginalization', False):
            from trajectory_radio_scale_mixture import scale_posterior
            return scale_posterior(z, A, self.device_prior, ld, self.prior)
        return mixture_posterior(z, A, self.device_prior, ld)

    def solve_axis(self, alternative, modes):
        predictions, diagnostic = ForceRawHMMPath.solve_axis(self, alternative, modes)
        return self.calibrate_solution(predictions, diagnostic)

    def calibrate_solution(self, predictions, diagnostic):
        q, _ = self.unpack(self.last_selected_vector[:-1])
        model = self.observations
        raw_xy = self.raw_spline @ q
        if hasattr(self, 'initial_dependency'):
            prefix_xy = self.initial_dependency_basis @ q
            self.device_prior = self.initial_dependency.at(prefix_xy)
            diagnostic.update(initial_position_device_dependency_strength=self.initial_dependency.strength, initial_dependency_predicted_prefix_xy_m=prefix_xy.tolist(), initial_dependency_timestamp_s=float(self.t[5]), initial_dependency_target_positions_used=False, initial_dependency_native_position_marginal_used=False, initial_dependency_recounts_initial_RSSI=False, initial_dependency_conditional_mixture_approximation=True)
        mean, basis = model.geometry(raw_xy, self.radio_field)
        z, A, ld = model.whiten(mean, basis, self.radio_blocks)
        _, eta, P, _, component_weight = self.device_posterior(z, A, ld, self.radio_blocks)
        if self.prior.get('RSSI_AP_scale_marginalization', False):
            from trajectory_AP_noise_mixture import ap_posterior
            details = ap_posterior(z, A, self.device_prior, ld, self.radio_blocks, self.prior, True)[-1]
            diagnostic.update(RSSI_AP_high_noise_posterior=details['AP_high_noise_posterior'].tolist(), RSSI_AP_noise_family_posterior=details['family_posterior'].tolist(), RSSI_AP_noise_families_integrated=True, RSSI_AP_noise_normalizers_retained=True, RSSI_AP_noise_scope='One scale per AP for the complete offline segment; no AP observations deleted')
        if self.prior.get('RSSI_segment_scale_marginalization', False):
            from trajectory_radio_scale_mixture import scale_posterior
            full_scale = scale_posterior(z, A, self.device_prior, ld, self.prior, True)[-1]['scale_posterior']
            diagnostic.update(RSSI_segment_scale_posterior=full_scale.tolist(), RSSI_segment_scale_factors=self.prior['RSSI_segment_scale_factors'])
        _, initial_mu, initial_P, _, _ = self.device_posterior(np.zeros(0), np.zeros((0, 6)))
        N = len(mean)
        means = np.broadcast_to(initial_mu, (N, 6)).copy()
        covariances = np.broadcast_to(initial_P, (N, 6, 6)).copy()
        delta = np.zeros((N, 4))
        raw_cells = [tuple(cell) for cell in np.floor(raw_xy / 2.0).astype(int)]
        now_cells = np.floor(self.B @ q / 2.0).astype(int)
        groups = {}
        for i in range(5, N):
            regions = {tuple(now_cells[i])}
            for ap in range(4):
                for j in np.where(model.S[ap][i] > 0)[0]:
                    regions.add(raw_cells[model.bounds[ap] + j])
            groups.setdefault(tuple(sorted(regions)), []).append(i)
        for regions, rows in groups.items():
            mask = self.pre['mask'].copy()
            region_set = set(regions)
            for ap in range(4):
                bad = np.asarray([raw_cells[j] in region_set for j in range(model.bounds[ap], model.bounds[ap + 1])])
                mask[model.S[ap][:, bad].sum(-1) > 0, ap] = False
            blocks = model.blocks(self.radio_covariances, mask)
            z, A, ld = model.whiten(mean, basis, blocks)
            _, mu, cov, _, _ = self.device_posterior(z, A, ld, blocks)
            means[rows] = mu
            covariances[rows] = cov
            for ap, ix, L in blocks:
                residual = self.pre['rssi'][ix, ap] - mean[ix, ap] - basis[ix, ap] @ mu
                if hasattr(model, 'initial_noise'):
                    residual -= model.initial_noise(ix, ap, mu)
                score = cho_solve((L, True), residual, check_finite=False)
                delta[rows, ap] = self.radio_shared_covariances[ap][np.ix_(rows, ix)] @ score
            if hasattr(model, 'initial_noise'):
                for ap in range(4):
                    delta[rows, ap] += model.initial_noise(rows, ap, mu)
        self.calibration_forecasts = dict(device_mean=means, device_covariance=covariances, short_residual=delta, mode=0)
        if self.prior.get('field_context_override_root'):
            self.calibration_forecasts['field_motion'] = self.pre['motion'].copy()
            diagnostic['field_context_scope'] = 'Source-trained rolling context with immutable original initial-five device prior'
        if hasattr(model, 'fixed_context'):
            self.calibration_forecasts['field_motion'] = np.broadcast_to(self.pre['motion'][4], self.pre['motion'].shape).copy()
            diagnostic['field_context_timing'] = 'Immutable initial-five context; subsequent RSSI updates device posterior and propagation nuisance'
        diagnostic.update(joint_marginal_device_mean=eta.tolist(), joint_marginal_device_covariance=P.tolist(), initial_mixture_posterior_component_weights=component_weight.tolist(), device_prior_PRE_index=5, measured_RSSI_contributor_clock_used=True, shared_raw_RSSI_noise_modeled=True, rolling_initial_raw_sample_overlap_excluded=True, region_cavity_groups=len(groups), RSSI_FORECAST_timing='Offline raw-contributor spatial-region-cavity', known_phone_heading_used=False)
        if hasattr(model, 'initial_location_basis'):
            diagnostic.update(device_spatial_basis_strength=model.device_spatial_strength, device_spatial_scope='Initial-XY nonlinear AP correction retained with live measured context; only its additional spatial variation is tempered; common field gain remains free')
        if len(component_weight) > 8:
            diagnostic.update(initial_population_posterior_mass=float(component_weight[8:].sum()), initial_population_initial_eight_components_retained=True, source_field_common_gain_mean=float(1 + eta[1] / 50), source_field_common_gain_std=float(math.sqrt(max(P[1, 1], 0)) / 50))
        if hasattr(self, 'known_start_diagnostic'):
            diagnostic['known_start_radio_calibration'] = self.known_start_diagnostic
        if self.prior.get('geometry_locked_device_basis', False):
            diagnostic.update(device_basis_position_query='Known initial XY; measured causal context still varies', spatial_device_correction='One common source-field gain and context-dependent AP offsets', source_field_common_gain_mean=float(1 + eta[1] / 50), source_field_common_gain_std=float(math.sqrt(max(P[1, 1], 0)) / 50), position_dependent_learned_device_basis_frozen=True)
        return (predictions, diagnostic)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['fit', 'score'], required=True)
    for name in ['base', 'capture', 'queries', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--lock-spatial-basis', action='store_true')
    p.add_argument('--freeze-field-context', action='store_true')
    p.add_argument('--cohorts', nargs='+', default=['original_fixed20', 'AP_movement'])
    a = p.parse_args()
    fit(a) if a.stage == 'fit' else score(a)
if __name__ == '__main__':
    main()
