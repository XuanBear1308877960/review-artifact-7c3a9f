import math
import numpy as np
from scipy.linalg import solve_triangular, cho_solve
from trajectory_joint_marginal import mixture_posterior
from trajectory_temporal_radio import TemporalJointPath
from trajectory_receiver_angular_fit import feature

def augmented_prior(original, sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Angular response requires a positive source-fitted scale')
    K = len(original['mean'])
    mean = np.zeros((K, 8))
    mean[:, :6] = original['mean']
    cov = np.zeros((K, 8, 8))
    cov[:, :6, :6] = original['covariance']
    cov[:, 6:, 6:] = np.eye(2) * sigma * sigma
    precision = np.zeros_like(cov)
    precision[:, :6, :6] = original['precision']
    precision[:, 6:, 6:] = np.eye(2) / (sigma * sigma)
    vector = np.zeros((K, 8))
    vector[:, :6] = original['vector']
    return dict(mean=mean, covariance=cov, precision=precision, vector=vector, logdet=original['logdet'] + 4 * math.log(sigma), log_weight=original['log_weight'].copy())

class ReceiverAngularPath(TemporalJointPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any((self.prior.get(k) for k in ['known_start_radio_conditioning', 'RSSI_segment_scale_marginalization', 'RSSI_AP_scale_marginalization', 'initial_position_device_dependency_strength', 'geometry_locked_device_basis', 'frozen_M3_spatial_head_scale', 'RSSI_history_noise_source_selected', 'RSSI_raw_history_noise_source_selected', 'RSSI_map_variance_covariance', 'RSSI_spatial_residual_covariance', 'receiver_curvature_sigma_db'])):
            raise ValueError('Evaluate the angular-response mechanism independently of alternative noise/initial/M3 models')
        self.angular_prior = augmented_prior(self.device_prior, float(self.prior['receiver_angular_sigma_db']))

    @staticmethod
    def whiten_angular(phi, blocks):
        return np.concatenate([solve_triangular(L, phi[ix, ap], lower=True, check_finite=False) for ap, ix, L in blocks]) if blocks else np.zeros((0, 2))

    def angular_posterior(self, z, A, ld, phi, blocks):
        return mixture_posterior(z, np.column_stack((A, self.whiten_angular(phi, blocks))), self.angular_prior, ld)

    def radio(self, q, mode):
        model = self.observations
        raw_xy = self.raw_spline @ q
        mean, basis, dm, db = model.geometry(raw_xy, self.radio_field, True)
        phi, dphi = feature(model, raw_xy, self.radio_field, self.initial, True)
        z, A, ld = model.whiten(mean, basis, self.radio_blocks)
        full = np.column_stack((A, self.whiten_angular(phi, self.radio_blocks)))
        loss, eta, P, residual, components = mixture_posterior(z, full, self.angular_prior, ld)
        gfull = full @ P - residual[:, None] * eta[None]
        gm = np.zeros_like(mean)
        gb = np.zeros_like(basis)
        gp = np.zeros_like(phi)
        offset = 0
        for ap, ix, L in self.radio_blocks:
            end = offset + len(ix)
            gm[ix, ap] = -solve_triangular(L.T, residual[offset:end], lower=False, check_finite=False)
            gb[ix, ap] = solve_triangular(L.T, gfull[offset:end, :6], lower=False, check_finite=False)
            gp[ix, ap] = solve_triangular(L.T, gfull[offset:end, 6:], lower=False, check_finite=False)
            offset = end
        gradient = model.pullback(gm, gb, dm, db, self.raw_spline)
        raw_score = np.concatenate([model.S[ap].T @ gp[:, ap] for ap in range(4)])
        gradient += self.raw_spline.T @ np.einsum('mc,mcz->mz', raw_score, dphi)
        return (float(loss), gradient, (eta, P, components))

    def calibrate_solution(self, predictions, diagnostic):
        q, _ = self.unpack(self.last_selected_vector[:-1])
        model = self.observations
        raw_xy = self.raw_spline @ q
        mean, basis = model.geometry(raw_xy, self.radio_field)
        phi = feature(model, raw_xy, self.radio_field, self.initial)
        z, A, ld = model.whiten(mean, basis, self.radio_blocks)
        _, eta, P, _, weights = self.angular_posterior(z, A, ld, phi, self.radio_blocks)
        _, initial_mu, initial_P, _, _ = self.angular_posterior(np.zeros(0), np.zeros((0, 6)), 0.0, phi, [])
        N = len(mean)
        means = np.broadcast_to(initial_mu, (N, 8)).copy()
        covariances = np.broadcast_to(initial_P, (N, 8, 8)).copy()
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
            _, mu, cov, _, _ = self.angular_posterior(z, A, ld, phi, blocks)
            means[rows] = mu
            covariances[rows] = cov
            for ap, ix, L in blocks:
                residual = self.pre['rssi'][ix, ap] - mean[ix, ap] - basis[ix, ap] @ mu[:6] - phi[ix, ap] @ mu[6:]
                score = cho_solve((L, True), residual, check_finite=False)
                delta[rows, ap] = self.radio_shared_covariances[ap][np.ix_(rows, ix)] @ score
        metadata = model.query_metadata
        horizontal = np.zeros((N, 4, 2))
        valid = np.zeros((N, 4))
        for ap in range(4):
            horizontal[:, ap] = model.S[ap] @ metadata['raw_attitude_horizontal_' + str(ap)]
            valid[:, ap] = model.S[ap] @ metadata['raw_attitude_valid_' + str(ap)].astype(float)
        self.calibration_forecasts = dict(device_mean=means[:, :6], device_covariance=covariances[:, :6, :6], short_residual=delta, mode=0, receiver_angular_mean_db=means[:, 6:], receiver_angular_covariance_db2=covariances[:, 6:, 6:], receiver_angular_query_horizontal=horizontal, receiver_angular_query_valid_weight=valid, receiver_angular_initial_horizontal=metadata['attitude_start_horizontal'].copy())
        if hasattr(model, 'fixed_context'):
            self.calibration_forecasts['field_motion'] = np.broadcast_to(self.pre['motion'][4], self.pre['motion'].shape).copy()
        if self.prior.get('field_context_override_root'):
            self.calibration_forecasts['field_motion'] = self.pre['motion'].copy()
        diagnostic.update(joint_marginal_device_mean=eta[:6].tolist(), joint_marginal_device_covariance=P[:6, :6].tolist(), initial_mixture_posterior_component_weights=weights.tolist(), receiver_angular_mean_db=eta[6:].tolist(), receiver_angular_covariance_db2=P[6:, 6:].tolist(), receiver_angular_original_six_initial_components_unchanged=True, receiver_angular_forecast_coefficient_norm_quantiles_db=np.percentile(np.linalg.norm(means[:, 6:], axis=-1), [0, 5, 50, 95, 100]).tolist(), receiver_angular_mean_map_changed=False, device_prior_PRE_index=5, region_cavity_groups=len(groups), RSSI_FORECAST_timing='Offline measured-contributor spatial-region cavity, integrated isotropic antenna-response pair', absolute_phone_map_heading_used=False, phone_heading_used_as_motion=False, quaternion_sensor_timestamp_available=False, attitude_clock='Already-arrived IMU receipt no later than raw RSSI measurement; 0.5s freshness mask')
        return (predictions, diagnostic)
