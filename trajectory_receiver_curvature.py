import math
import numpy as np
from scipy.linalg import solve_triangular, cho_solve
from trajectory_joint_marginal import mixture_posterior
from trajectory_temporal_radio import TemporalJointPath

def augmented_prior(original, sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Receiver curvature needs a positive source-fitted prior scale')
    K = len(original['mean'])
    mean = np.zeros((K, 7))
    mean[:, :6] = original['mean']
    cov = np.zeros((K, 7, 7))
    cov[:, :6, :6] = original['covariance']
    cov[:, 6, 6] = sigma * sigma
    precision = np.zeros_like(cov)
    precision[:, :6, :6] = original['precision']
    precision[:, 6, 6] = 1 / (sigma * sigma)
    vector = np.zeros((K, 7))
    vector[:, :6] = original['vector']
    return dict(mean=mean, covariance=cov, precision=precision, vector=vector, logdet=original['logdet'] + 2 * math.log(sigma), log_weight=original['log_weight'].copy())

class ReceiverCurvaturePath(TemporalJointPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any((self.prior.get(k) for k in ['known_start_radio_conditioning', 'RSSI_segment_scale_marginalization', 'RSSI_AP_scale_marginalization', 'initial_position_device_dependency_strength', 'geometry_locked_device_basis', 'frozen_M3_spatial_head_scale', 'RSSI_history_noise_source_selected', 'RSSI_map_variance_covariance', 'RSSI_spatial_residual_covariance'])):
            raise ValueError('Evaluate the receiver-curvature mechanism independently of alternative initial/noise/M3 models')
        self.curvature_prior = augmented_prior(self.device_prior, float(self.prior['receiver_curvature_sigma_db']))
        self.curvature_start_mean = self.radio_field.source_ap(np.broadcast_to(self.initial, (4, 2)), np.arange(4))

    def curvature(self, raw_xy, raw_dm=None):
        model = self.observations
        raw_mean = self.radio_field.source_ap(raw_xy, model.ap)
        phi = ((raw_mean + 60) / 20) ** 2 - ((self.curvature_start_mean[model.ap] + 60) / 20) ** 2
        frame = np.zeros((model.N, 4))
        for ap in range(4):
            frame[:, ap] = model.S[ap] @ phi[model.bounds[ap]:model.bounds[ap + 1]]
        if raw_dm is None:
            return frame
        return (frame, ((raw_mean + 60) / 200)[:, None] * raw_dm)

    @staticmethod
    def whiten_curvature(phi, blocks):
        return np.concatenate([solve_triangular(L, phi[ix, ap], lower=True, check_finite=False) for ap, ix, L in blocks]) if blocks else np.zeros(0)

    def curvature_posterior(self, z, A, ld, phi, blocks):
        v = self.whiten_curvature(phi, blocks)
        return mixture_posterior(z, np.column_stack((A, v)), self.curvature_prior, ld)

    def radio(self, q, mode):
        model = self.observations
        raw_xy = self.raw_spline @ q
        mean, basis, dm, db = model.geometry(raw_xy, self.radio_field, True)
        phi, dphi = self.curvature(raw_xy, dm)
        z, A, ld = model.whiten(mean, basis, self.radio_blocks)
        v = self.whiten_curvature(phi, self.radio_blocks)
        full = np.column_stack((A, v))
        loss, eta, P, residual, components = mixture_posterior(z, full, self.curvature_prior, ld)
        gfull = full @ P - residual[:, None] * eta[None]
        gm = np.zeros_like(mean)
        gb = np.zeros_like(basis)
        gp = np.zeros_like(mean)
        offset = 0
        for ap, ix, L in self.radio_blocks:
            end = offset + len(ix)
            gm[ix, ap] = -solve_triangular(L.T, residual[offset:end], lower=False, check_finite=False)
            gb[ix, ap] = solve_triangular(L.T, gfull[offset:end, :6], lower=False, check_finite=False)
            gp[ix, ap] = solve_triangular(L.T, gfull[offset:end, 6], lower=False, check_finite=False)
            offset = end
        gradient = model.pullback(gm, gb, dm, db, self.raw_spline)
        raw_score = np.concatenate([model.S[ap].T @ gp[:, ap] for ap in range(4)])
        gradient += self.raw_spline.T @ (raw_score[:, None] * dphi)
        return (float(loss), gradient, (eta, P, components))

    def calibrate_solution(self, predictions, diagnostic):
        q, _ = self.unpack(self.last_selected_vector[:-1])
        model = self.observations
        raw_xy = self.raw_spline @ q
        mean, basis = model.geometry(raw_xy, self.radio_field)
        phi = self.curvature(raw_xy)
        z, A, ld = model.whiten(mean, basis, self.radio_blocks)
        _, eta, P, _, weights = self.curvature_posterior(z, A, ld, phi, self.radio_blocks)
        _, initial_mu, initial_P, _, _ = self.curvature_posterior(np.zeros(0), np.zeros((0, 6)), 0.0, phi, [])
        N = len(mean)
        means = np.broadcast_to(initial_mu, (N, 7)).copy()
        covariances = np.broadcast_to(initial_P, (N, 7, 7)).copy()
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
            _, mu, cov, _, _ = self.curvature_posterior(z, A, ld, phi, blocks)
            means[rows] = mu
            covariances[rows] = cov
            for ap, ix, L in blocks:
                residual = self.pre['rssi'][ix, ap] - mean[ix, ap] - basis[ix, ap] @ mu[:6] - phi[ix, ap] * mu[6]
                score = cho_solve((L, True), residual, check_finite=False)
                delta[rows, ap] = self.radio_shared_covariances[ap][np.ix_(rows, ix)] @ score
        self.calibration_forecasts = dict(device_mean=means[:, :6], device_covariance=covariances[:, :6, :6], short_residual=delta, mode=0, receiver_curvature_mean_db=means[:, 6], receiver_curvature_variance_db2=covariances[:, 6, 6], receiver_curvature_initial_mean_db=self.curvature_start_mean.copy())
        if hasattr(model, 'fixed_context'):
            self.calibration_forecasts['field_motion'] = np.broadcast_to(self.pre['motion'][4], self.pre['motion'].shape).copy()
        if self.prior.get('field_context_override_root'):
            self.calibration_forecasts['field_motion'] = self.pre['motion'].copy()
        diagnostic.update(joint_marginal_device_mean=eta[:6].tolist(), joint_marginal_device_covariance=P[:6, :6].tolist(), initial_mixture_posterior_component_weights=weights.tolist(), receiver_curvature_mean_db=float(eta[6]), receiver_curvature_variance_db2=float(P[6, 6]), receiver_curvature_original_six_initial_components_unchanged=True, receiver_curvature_mean_map_changed=False, receiver_curvature_prior_scope='Independent scalar prior conditional on frozen original recognition; no refitting or recounting initial-five observations', receiver_curvature_forecast_mean_quantiles_db=np.percentile(means[:, 6], [0, 5, 50, 95, 100]).tolist(), receiver_curvature_forecast_sigma_quantiles_db=np.percentile(np.sqrt(np.maximum(covariances[:, 6, 6], 0)), [0, 5, 50, 95, 100]).tolist(), device_prior_PRE_index=5, measured_RSSI_contributor_clock_used=True, shared_raw_RSSI_noise_modeled=True, rolling_initial_raw_sample_overlap_excluded=True, region_cavity_groups=len(groups), RSSI_FORECAST_timing='Offline raw-contributor spatial-region-cavity with integrated receiver curvature', known_phone_heading_used=False)
        return (predictions, diagnostic)
