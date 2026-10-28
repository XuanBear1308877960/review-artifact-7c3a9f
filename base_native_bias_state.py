from __future__ import annotations
import copy
import math
from types import SimpleNamespace
import numpy as np
import torch
from base_model import gaussian_log_prob, stabilize
from base_data import radio_tokens
from base_quadrature import position_support
from base_radio_likelihood import robust_radio_score
from base_spatial_geometry import calibrated_field
from base_cs_temporal import configure_filter, replace_device

class NativeEngine:

    def __init__(self, net, aps, anchors, initial_xy, initial_ns, filter_cfg, map_refine=False, cs_iterations=1, position_rule='mixed'):
        from support_motion_context import CausalMotionContext
        from base_v46_motion import MotionPredictor
        self.net = net
        self.aps = list(aps)
        self.lookup = {a: i for i, a in enumerate(aps)}
        self.anchor_names = list(anchors)
        self.device = next(net.parameters()).device
        self.anchors = torch.tensor(list(anchors.values()), device=self.device, dtype=torch.float32)
        self.k = 8
        self.a = len(aps)
        self.timestamp = int(initial_ns)
        self.heading = torch.arange(8, device=self.device) * math.pi / 4
        self.filter_cfg = configure_filter(filter_cfg, net.cs_temporal_noise, self.anchor_names)
        self.motion = MotionPredictor()
        self.temporal = self.filter_cfg.get('cs_temporal_noise')
        if self.temporal and list(self.temporal) != self.anchor_names:
            raise ValueError('CS temporal anchor order differs')
        self.dim = 19 if self.temporal else 15
        self.eta = slice(9, 15)
        self.eta_u = slice(7, 13)
        self.map_refine = map_refine
        self.solver_diagnostics = []
        if map_refine and net.robust_radio:
            raise ValueError('Robust radio uses mixture integration; MAP64 is a separate research mode')
        self.cs_iterations = int(cs_iterations)
        self.position_rule = position_rule
        self.grid_xy = net.position_nodes()
        self.cell_rule = net.integration_grid.kind != 'legacy'
        if self.cell_rule and (position_rule != 'mixed' or map_refine or (not net.aligned_motion)):
            raise ValueError('Source-cell quadrature uses the aligned mixed Gaussian-sum filter')
        self.rotation = None
        self.first_rotation = None
        self.acceleration = np.zeros(2)
        if net.temporal_contrast and map_refine:
            raise ValueError('Temporal contrast is integrated with the native Gaussian-sum position update')
        self.motion_total = torch.zeros(self.k, 2, device=self.device)
        self.motion_paths = []
        self.delta_history = []
        self.motion_observer = CausalMotionContext(aps)
        self.motion_tokens = []
        mu, cov = net.calibration.prior((self.k,))
        self.mean = torch.zeros(self.k, self.dim, device=self.device)
        self.mean[:, :2] = torch.tensor(initial_xy, device=self.device)
        self.mean[:, self.eta] = mu
        self.mean[:, 4] = self.heading
        self.cov = torch.zeros(self.k, self.dim, self.dim, device=self.device)
        self.cov[:, 2:4, 2:4] = torch.eye(2, device=self.device)
        self.cov[:, 4, 4] = (math.pi / 8) ** 2 / 3
        self.cov[:, 5:9, 5:9] = torch.eye(4, device=self.device) * filter_cfg['initial_bias_sigma_m'] ** 2
        self.cov[:, self.eta, self.eta] = cov
        if self.temporal:
            self.cov[:, 15:, 15:] = torch.diag(torch.tensor([p['stationary_sigma_m'] ** 2 for p in self.temporal.values()], device=self.device))
        self.log_weight = torch.full((self.k,), -math.log(self.k), device=self.device)
        self.initial_information = torch.zeros(self.k, 6, 6, device=self.device)
        self.initial_vector = torch.zeros(self.k, 6, device=self.device)
        self.initial_ids = []
        self.initial_device_posterior = None
        self.initial_window = []
        self.initial_mass = torch.zeros(self.k, self.a, device=self.device)
        self.initial_basis_sum = torch.zeros(self.k, self.a, 6, device=self.device)
        self.initial_offset_sum = torch.zeros_like(self.initial_mass)
        self.initial_noise_sum = torch.zeros_like(self.initial_mass)
        self.retired_information = torch.zeros_like(self.initial_information)
        self.retired_vector = torch.zeros_like(self.initial_vector)
        self.regions = [self._region(initial_xy)]
        self.radio = []
        self.innovations = []
        self.predicted_poses = []
        self.pose_variances = []
        self.innovation_designs = []
        self.innovation_device_means = []
        self.times = []
        self.rssi_history = []
        self.mask_history = []
        self.cs_history = {a: [] for a in self.anchor_names}
        self.wifi_axes = []
        self.ledger = set()
        self.forecasts = []
        self.outputs = []
        self.cs_diagnostics = []
        self.radio_diagnostics = []
        self.cs_training_rows = []
        self.last_wifi_distribution = None
        self.current_motion = torch.zeros(32, device=self.device)
        with torch.no_grad():
            self.grid_fields = net.field.source(self.grid_xy)
            if net.field.source_model == 'idw':
                gradient = []
                for axis in range(2):
                    d = torch.zeros_like(self.grid_xy)
                    d[:, axis] = 0.15
                    gradient.append((net.field.source(self.grid_xy + d)['mean'] - net.field.source(self.grid_xy - d)['mean']) / 0.3)
                self.grid_fields['gradient'] = torch.stack(gradient, -1)

    def _region(self, origin):
        return dict(information=torch.zeros(self.k, 6, 6, device=self.device), vector=torch.zeros(self.k, 6, device=self.device), origin=np.asarray(origin, float), original_probability=torch.ones(self.k, device=self.device), observation_ids=[], count=0, transfer_mean=None, transfer_covariance=None, entry_offset=None)

    def _advance(self, ns):
        dt = (int(ns) - self.timestamp) / 1000000000.0
        if dt < 0:
            raise ValueError('Noncausal event order')
        if dt == 0:
            return
        mean, covariance = self.motion.predict_batch(self.mean.cpu().numpy(), self.cov.cpu().numpy(), dt, self.filter_cfg)
        previous_position = self.mean[:, :2]
        self.mean = torch.tensor(mean, device=self.device, dtype=torch.float32)
        if self.net.temporal_contrast:
            self.motion_total = self.motion_total + (self.mean[:, :2] - previous_position)
        self.cov = torch.tensor(covariance, device=self.device, dtype=torch.float32)
        self.motion.advance_reference(dt, self.filter_cfg)
        self.timestamp = int(ns)

    def _motion_token(self, event):
        proxy = SimpleNamespace(**event)
        if event['kind'] == 'wifi':
            token = self.motion_observer.consume(proxy)
        else:
            observer = copy.deepcopy(self.motion_observer)
            proxy = SimpleNamespace(**dict(event, kind='wifi', payload={'ap_observations': []}))
            token = observer.consume(proxy)
        if event['kind'] == 'wifi' or not self.net.aligned_motion:
            self.motion_tokens.append(token)
            values = np.asarray(self.motion_tokens[-16:], np.float32)
        else:
            values = np.asarray(self.motion_tokens[-15:] + [token], np.float32)
        if self.net.aligned_motion:
            values = np.pad(values, ((16 - len(values), 0), (0, 0)))
        self.current_motion_history = values
        history = torch.tensor(values, device=self.device)
        self.current_motion = torch.tanh(self.net.calibration.motion_encoder(history)[-1])

    def _cs_features(self, anchor):
        seq = self.cs_history[anchor]
        a = self.anchor_names.index(anchor)
        t, raw, corrected, oid = seq[-1]
        dt = t - seq[-2][0] if len(seq) > 1 else 0.0
        valid_dt = 0.0001 < dt < 10
        change = corrected - seq[-2][2] if valid_dt else 0.0
        recent = [v for v in seq[-9:] if t - v[0] <= 10]
        values = np.asarray([v[2] for v in recent])
        median = float(np.median(values))
        features = [raw / 20, corrected / 20, (corrected - raw) / 10, change / 10, abs(change) / 10, dt / 5, change / max(dt, 0.01) / 10, median / 20, float(np.median(abs(values - median))) / 5, len(recent) / 9, (recent[-1][0] - recent[0][0]) / 10, 0.0, 1.0, float(len(seq) > 1), float(valid_dt), 1.0, *[float(a == j) for j in range(self.a)]]
        return torch.tensor(features, device=self.device, dtype=torch.float32)

    def _consume_cs(self, event):
        p = event['payload']
        anchor = p['anchor_id']
        if anchor not in self.cs_history:
            return
        now = self.timestamp / 1000000000.0
        self.cs_history[anchor].append((now, float(p['raw_range_m']), float(p['range_m']), event['observation_id']))
        feature = self._cs_features(anchor)
        if self.net.aligned_motion:
            index = 5 + self.anchor_names.index(anchor)
            self.cs_training_rows.append(dict(observation_id=event['observation_id'], timestamp_ns=self.timestamp, features=feature.cpu().numpy(), motion_history=self.current_motion_history.copy(), range_m=float(p['range_m']), bias_mean=self.mean[:, index].cpu().numpy().copy(), bias_variance=self.cov[:, index, index].cpu().numpy().copy(), mode_weights=self.log_weight.softmax(-1).cpu().numpy().copy(), velocity=self.mean[:, 2:4].cpu().numpy().copy(), anchor_index=self.anchor_names.index(anchor), wifi_index=len(self.forecasts) - 1, wifi_age_s=1000.0 if not self.forecasts else (self.timestamp - self.forecasts[-1]['timestamp_ns']) / 1000000000.0))
            if self.temporal:
                noise_index = 15 + self.anchor_names.index(anchor)
                self.cs_training_rows[-1].update(temporal_mean=self.mean[:, noise_index].cpu().numpy().copy(), temporal_variance=self.cov[:, noise_index, noise_index].cpu().numpy().copy(), bias_temporal_covariance=self.cov[:, index, noise_index].cpu().numpy().copy())
        sigma, pi = self.net.calibration.cs_parameters(feature, self.current_motion)
        a = self.anchors[self.anchor_names.index(anchor)]
        displacement = self.mean[:, :2] - a
        distance = displacement.square().sum(-1).clamp_min(1e-08).sqrt()
        index = 5 + self.anchor_names.index(anchor)
        additive = self.mean[:, index]
        additive_variance = self.cov[:, index, index]
        noise_index = 15 + self.anchor_names.index(anchor) if self.temporal else None
        if noise_index is not None:
            additive = additive + self.mean[:, noise_index]
            additive_variance = additive_variance + self.cov[:, noise_index, noise_index] + 2 * self.cov[:, index, noise_index]
        predicted = distance + additive
        h = torch.zeros(self.k, self.dim, device=self.device)
        h[:, :2] = displacement / distance[:, None]
        h[:, index] = 1.0
        if noise_index is not None:
            h[:, noise_index] = 1.0
        hp = (h[:, None] @ self.cov).squeeze(1)
        predicted_variance = (hp * h).sum(-1).clamp_min(0.0)
        residual = float(p['range_m']) - predicted
        s = sigma.square() + 0.15 ** 2 + predicted_variance
        nominal = -0.5 * (residual.square() / s + s.log() + math.log(2 * math.pi))
        tail = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi) - math.log(10.0) - 2 * torch.log1p(residual.square() / 300)
        nominal = nominal + torch.log1p(-pi)
        tail = tail + pi.log()
        evidence = torch.logaddexp(nominal, tail)
        responsibility = (nominal - evidence).exp()
        state_responsibility = responsibility.clone()
        feedback_used = False
        if self.last_wifi_distribution is not None and self.filter_cfg.get('cs_radio_feedback_policy') != 'state_only':
            old = self.last_wifi_distribution
            age = now - old['time']
            if age <= 1.5:
                shift = self.mean[:, None, 2:4] * age
                location = old['points'] + shift
                predicted = (location - a).square().sum(-1).clamp_min(1e-08).sqrt() + additive[:, None]
                error = float(p['range_m']) - predicted
                uncertainty = sigma.square() + 0.15 ** 2 + 4 * max(age, 0.0) ** 3 / 3 + additive_variance.clamp_min(0.0)[:, None]
                normal = -0.5 * (error.square() / uncertainty + uncertainty.log() + math.log(2 * math.pi))
                heavy = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi) - math.log(10.0) - 2 * torch.log1p(error.square() / 300)
                weights = old['probability'].clamp_min(1e-20).log()
                normal = torch.logsumexp(weights + normal, -1) + torch.log1p(-pi)
                heavy = torch.logsumexp(weights + heavy, -1) + pi.log()
                responsibility = (normal - torch.logaddexp(normal, heavy)).exp()
                feedback_used = True
        noise = (sigma.square() + 0.15 ** 2) / responsibility.clamp_min(0.001)
        termination = ['SINGLE_LINEARIZATION'] * self.k
        iteration_counts = [1] * self.k
        if self.cs_iterations == 1:
            denominator = predicted_variance + noise
            gain = hp / denominator[:, None]
            self.mean = self.mean + gain * residual[:, None]
            self.mean[:, 5:9] = self.mean[:, 5:9].clamp(-self.filter_cfg['bias_limit_m'], self.filter_cfg['bias_limit_m'])
        else:
            original = self.mean.clone()
            prior = self.cov
            prior_precision = torch.linalg.inv(stabilize(prior))

            def linearize(x):
                diff = x[:, :2] - a
                d = diff.square().sum(-1).clamp_min(1e-08).sqrt()
                jac = torch.zeros_like(x)
                jac[:, :2] = diff / d[:, None]
                jac[:, index] = 1.0
                value = d + x[:, index]
                if noise_index is not None:
                    jac[:, noise_index] = 1.0
                    value = value + x[:, noise_index]
                return (value, jac)

            def objective(x):
                value, _ = linearize(x)
                change = x - original
                return (float(p['range_m']) - value).square() / noise + torch.einsum('kd,kde,ke->k', change, prior_precision, change)
            x = original.clone()
            cost = objective(x)
            running = torch.ones(self.k, dtype=torch.bool, device=self.device)
            termination = ['ITERATION_LIMIT'] * self.k
            for iteration in range(self.cs_iterations):
                pred, h = linearize(x)
                hp = (h[:, None] @ prior).squeeze(1)
                gain = hp / ((hp * h).sum(-1) + noise)[:, None]
                innovation = float(p['range_m']) - pred + (h * (x - original)).sum(-1)
                proposal = original + gain * innovation[:, None]
                proposal[:, 5:9] = proposal[:, 5:9].clamp(-self.filter_cfg['bias_limit_m'], self.filter_cfg['bias_limit_m'])
                accepted = torch.zeros_like(running)
                updated = x.clone()
                updated_cost = cost.clone()
                for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
                    trial = x + alpha * (proposal - x)
                    trial_cost = objective(trial)
                    choose = running & ~accepted & (trial_cost <= cost + 1e-07)
                    updated = torch.where(choose[:, None], trial, updated)
                    updated_cost = torch.where(choose, trial_cost, updated_cost)
                    accepted |= choose
                step = (updated - x).square().sum(-1).sqrt()
                for j in range(self.k):
                    if bool(running[j]):
                        iteration_counts[j] = iteration + 1
                        if not bool(accepted[j]):
                            termination[j] = 'LINE_SEARCH_STALL'
                        elif float(step[j]) < 0.0001:
                            termination[j] = 'STEP_TOLERANCE'
                x = updated
                cost = updated_cost
                running &= accepted & (step >= 0.0001)
                if not bool(running.any()):
                    break
            self.mean = x
            _, h = linearize(x)
            hp = (h[:, None] @ prior).squeeze(1)
            gain = hp / ((hp * h).sum(-1) + noise)[:, None]
        transform = torch.eye(self.dim, device=self.device)[None] - gain[..., None] * h[:, None]
        self.cov = stabilize(transform @ self.cov @ transform.transpose(-1, -2) + noise[:, None, None] * gain[..., None] * gain[:, None])
        self.log_weight = (self.log_weight + evidence).log_softmax(-1)
        self.cs_diagnostics.append(dict(observation_id=event['observation_id'], sigma=float(sigma), outlier_probability=float(pi), nominal_responsibility=responsibility.tolist(), prior_state_nominal_responsibility=state_responsibility.tolist(), cs_feedback_policy=self.filter_cfg.get('cs_radio_feedback_policy', 'wifi_map'), wifi_feedback=feedback_used, measurement_consumed_once=True, solver_iterations=iteration_counts, solver_termination=termination))
        if self.temporal:
            self.cs_diagnostics[-1].update(sigma_semantics='Conditional white CS noise; OU state is separate', temporal_mean=self.mean[:, noise_index].tolist(), temporal_variance=self.cov[:, noise_index, noise_index].tolist())

    def _device_cavity(self):
        if self.net.initial_posterior != 'none' and self.initial_device_posterior is not None:
            information, vector = self._device_region_factors()
            return self.net.calibration.posterior(information, vector, *self.initial_device_posterior)
        information = self.initial_information + self.retired_information
        vector = self.initial_vector + self.retired_vector
        for region in self.regions[:-1]:
            information = information + region['information']
            vector = vector + region['vector']
        return self.net.calibration.posterior(information, vector)

    def _device_region_factors(self):
        if self.net.region_reliability != 'none':
            from base_region_reliability import closed_factors
            information, vector, probability = self._closed_region_bank()
            base = self.initial_device_posterior if self.initial_device_posterior is not None else self.net.calibration.prior((self.k,))
            information, vector, _ = closed_factors(self.net.calibration, self.net.region_reliability, information, vector, probability, self.retired_information, self.retired_vector, *base)
            return (information, vector)
        information = self.retired_information.clone()
        vector = self.retired_vector.clone()
        for region in self.regions[:-1]:
            information += region['information']
            vector += region['vector']
        return (information, vector)

    def _closed_region_bank(self):
        information = torch.zeros(self.k, 3, 6, 6, device=self.device)
        vector = torch.zeros(self.k, 3, 6, device=self.device)
        probability = torch.zeros(self.k, 3, device=self.device)
        old = self.regions[:-1]
        for j, region in enumerate(old, 3 - len(old)):
            information[:, j] = region['information']
            vector[:, j] = region['vector']
            probability[:, j] = region['original_probability']
        return (information, vector, probability)

    def _device_prediction(self):
        if self.net.online_device_policy == 'cavity':
            return self._device_cavity()
        information = self.retired_information.clone()
        vector = self.retired_vector.clone()
        if self.initial_device_posterior is None:
            information = information + self.initial_information
            vector = vector + self.initial_vector
        for region in self.regions:
            information = information + region['information']
            vector = vector + region['vector']
        return self.net.calibration.posterior(information, vector, *(self.initial_device_posterior if self.initial_device_posterior is not None else (None, None)))

    def _encoded(self, mask):
        h = np.asarray(self.radio[-16:], np.float32)
        if len(h) == 0:
            h = np.zeros((0, self.a, 8), np.float32)
        h = np.pad(h, ((16 - len(h), 0), (0, 0), (0, 0))).transpose(1, 0, 2)
        initial = np.asarray(self.radio[:5], np.float32)
        if len(initial) == 0:
            initial = np.zeros((0, self.a, 8), np.float32)
        initial = np.pad(initial, ((5 - len(initial), 0), (0, 0), (0, 0))).transpose(1, 0, 2)
        motion = np.asarray(self.motion_tokens[-16:], np.float32)
        motion = np.pad(motion, ((16 - len(motion), 0), (0, 0)))
        return self.net.calibration.encode_history(torch.tensor(h[None], device=self.device), torch.tensor(motion[None], device=self.device), torch.tensor(initial[None], device=self.device), mask[None])

    def _short_residual(self, encoded, device_mean=None):
        if not self.innovations:
            return torch.zeros(self.k, self.a, device=self.device)
        n = min(16, len(self.innovations))
        innovation = torch.stack(self.innovations[-n:])
        if self.net.memory_recenter:
            if device_mean is None:
                raise ValueError('Residual recentering requires the current causal device mean')
            design = torch.stack(self.innovation_designs[-n:])
            previous = torch.stack(self.innovation_device_means[-n:])
            innovation = innovation - torch.einsum('hkad,hkd->hka', design, device_mean[None] - previous)
        innovation = innovation.permute(1, 2, 0)[None]
        valid = torch.tensor(np.asarray(self.mask_history[-n:]), device=self.device).T[None, None].expand(1, self.k, self.a, n)
        ages = torch.tensor(self.timestamp / 1000000000.0 - np.asarray(self.times[-n:]), device=self.device, dtype=torch.float32)[None, None].expand(1, self.k, n)
        poses = torch.stack(self.predicted_poses[-n:]).transpose(0, 1)
        variance = torch.stack(self.pose_variances[-n:]).T
        current = self.cov[:, :2, :2].diagonal(dim1=-2, dim2=-1).sum(-1)
        spatial = torch.exp(-0.5 * (poses - self.mean[:, None, :2]).square().sum(-1) / (1 + current[:, None] + variance))
        confidence = (1 + current / 4).rsqrt()[None]
        gap = torch.tensor([self.timestamp / 1000000000.0 - self.times[-1]], device=self.device)
        axes = np.asarray(self.wifi_axes[-n:])
        current = np.asarray(self.motion_tokens[-1])[[0, 2]]
        cosine = (axes * current).sum(-1) / (np.linalg.norm(axes, axis=-1) * np.linalg.norm(current)).clip(1e-06)
        rotation = torch.tensor(np.exp(2 * (cosine.clip(-1, 1) - 1)), device=self.device, dtype=torch.float32)[None, None].expand(1, self.k, n)
        present = sum((bool(seq and self.timestamp / 1000000000.0 - seq[-1][0] <= 1.5) for seq in self.cs_history.values())) / self.a
        delta, _ = self.net.calibration.short_residual(encoded, self.net.field.ap_embedding.weight, innovation, ages, valid, confidence, spatial[None], rotation, gap, torch.tensor([present], device=self.device))
        return delta[0]

    def _point_score(self, points, encoded, mu, cov, delta, y, mask):
        field = self.net.field(points, encoded['motion'][0].expand(self.k, 32), heading=self.mean[:, 4])
        field = calibrated_field(field, mu)
        basis = field['basis']
        prediction = field['mean'] + (basis @ mu[..., None]).squeeze(-1) + delta
        noise = field['map_variance'] + encoded['sigma'][0].square()
        s = basis @ cov @ basis.transpose(-1, -2) + torch.diag_embed(noise)
        m = mask.float()
        s = s * m[:, None] * m[None, :] + torch.diag(1 - m)
        score, chol = gaussian_log_prob((y - prediction) * m, stabilize(s), return_factor=True)
        score = score + 0.5 * (~mask).sum() * math.log(2 * math.pi)
        dv = (basis @ cov @ basis.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1)
        score = score + self.net.localization(y.expand(self.k, self.a), mask.expand(self.k, self.a), prediction, s.diagonal(dim1=-2, dim2=-1), field, dv, self.net.field.ap_embedding.weight, encoded['spatial_radio'][0].expand(self.k, self.a, 24), encoded['spatial_motion'][0].expand(self.k, 32), s, chol)
        if self.net.robust_radio:
            score, _ = robust_radio_score(score, y, mask, prediction, s, encoded['outlier'][0])
        return score

    def _refine_position(self, start, prior_mean, prior_cov, encoded, mu, cov, delta, y, mask, oid):
        precision = torch.linalg.inv(prior_cov)

        def cost(points):
            difference = points - prior_mean
            return 0.5 * torch.einsum('kd,kde,ke->k', difference, precision, difference) - self._point_score(points, encoded, mu, cov, delta, y, mask)

        def derivative(points):
            with torch.enable_grad():
                query = points.detach().requires_grad_(True)
                value = cost(query)
                gradient = torch.autograd.grad(value.sum(), query, create_graph=False)[0]
            return (value.detach(), gradient.detach())
        before = cost(prior_mean).detach()
        initial_value = cost(start).detach()
        x = torch.where((initial_value <= before)[:, None], start, prior_mean).detach()
        value, gradient = derivative(x)
        inverse_hessian = prior_cov.clone()
        active = torch.ones(self.k, dtype=torch.bool, device=self.device)
        iterations = torch.zeros(self.k, dtype=torch.long, device=self.device)
        reasons = ['ITERATION_LIMIT'] * self.k
        for _ in range(64):
            if not bool(active.any()):
                break
            step = -(inverse_hessian @ gradient[..., None]).squeeze(-1)
            mahal = torch.einsum('kd,kde,ke->k', step, precision, step).clamp_min(1e-12).sqrt()
            step = step * (3 / mahal).clamp_max(1.0)[:, None]
            candidate = x.clone()
            accepted = torch.zeros_like(active)
            for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
                trial = x + alpha * step
                trial_value = cost(trial).detach()
                take = active & ~accepted & torch.isfinite(trial_value) & (trial_value <= value + 1e-07)
                candidate = torch.where(take[:, None], trial, candidate)
                accepted |= take
                if bool((accepted | ~active).all()):
                    break
            next_value, next_gradient = derivative(candidate)
            displacement = candidate - x
            change = next_gradient - gradient
            curvature = (displacement * change).sum(-1)
            usable = accepted & (curvature > 1e-08)
            rho = torch.where(usable, curvature.clamp_min(1e-08).reciprocal(), torch.zeros_like(curvature))
            eye = torch.eye(2, device=self.device)[None]
            transform = eye - rho[:, None, None] * displacement[..., None] * change[:, None]
            updated = transform @ inverse_hessian @ transform.transpose(-1, -2) + rho[:, None, None] * displacement[..., None] * displacement[:, None]
            inverse_hessian = torch.where(usable[:, None, None], stabilize(updated), prior_cov)
            small = displacement.square().sum(-1).sqrt() < 0.0001
            for j in range(self.k):
                if bool(active[j]):
                    iterations[j] += 1
                    if not bool(accepted[j]):
                        reasons[j] = 'LINE_SEARCH_STALL'
                    elif bool(small[j]):
                        reasons[j] = 'STEP_TOLERANCE'
            active = active & accepted & ~small
            x, value, gradient = (candidate, next_value, next_gradient)
        self.solver_diagnostics.append(dict(observation_id=oid, iterations=iterations.tolist(), termination=reasons, prior_cost=before.tolist(), final_cost=value.tolist(), original_prior_reused=True, network_weights_updated=False))
        return x

    def _temporal_context(self, delta):

        def pad(values, shape, dtype=torch.float32):
            selected = values[-16:]
            data = torch.stack([torch.as_tensor(x, device=self.device, dtype=dtype) for x in selected]) if selected else torch.zeros(0, *shape, device=self.device, dtype=dtype)
            return torch.cat((torch.zeros(16 - len(selected), *shape, device=self.device, dtype=dtype), data))[None]
        now = self.timestamp / 1000000000.0
        times = np.r_[np.full(16 - min(16, len(self.times)), -1000.0), np.asarray(self.times[-16:]) - now]
        return dict(rssi=pad(self.rssi_history, (self.a,)), mask=pad(self.mask_history, (self.a,), torch.bool), delta=pad(self.delta_history, (self.k, self.a)), path=pad(self.motion_paths, (self.k, 2)), variance=pad(self.pose_variances, (self.k,)), axes=pad(self.wifi_axes, (2,)), times=torch.tensor(times, device=self.device, dtype=torch.float32)[None], now_time=torch.zeros(1, device=self.device), now_path=self.motion_total[None], now_variance=self.cov[:, :2, :2].diagonal(dim1=-2, dim2=-1).sum(-1)[None], now_axis=torch.tensor(np.asarray(self.motion_tokens[-1])[[0, 2]], device=self.device)[None])

    def _consume_wifi(self, event):
        y = torch.zeros(self.a, device=self.device)
        mask = torch.zeros(self.a, dtype=torch.bool, device=self.device)
        for ap in event['payload']['ap_observations']:
            a = self.lookup.get(ap['ap_id_hash'])
            if a is not None:
                y[a] = float(ap['rssi_dbm'])
                mask[a] = True
        if not mask.any():
            return
        encoded = self._encoded(mask)
        mu, cov = self._device_prediction()
        delta = self._short_residual(encoded, mu)
        temporal_context = self._temporal_context(delta) if self.net.temporal_contrast else None
        region = self.regions[-1]
        if self.net.mode_evidence != 'local' and region['transfer_mean'] is None:
            outside_mean, outside_covariance = self._device_cavity()
            region['transfer_mean'] = outside_mean.clone()
            region['transfer_covariance'] = outside_covariance.clone()
            region['entry_offset'] = delta.clone()
        forecast = dict(observation_id=event['observation_id'], timestamp_ns=self.timestamp, device_mean=mu.cpu().numpy(), device_covariance=cov.cpu().numpy(), short_residual=delta.cpu().numpy(), motion=encoded['motion'][0].cpu().numpy(), weights=self.log_weight.softmax(-1).cpu().numpy(), rssi=y.cpu().numpy(), mask=mask.cpu().numpy(), pre_mean=self.mean.cpu().numpy().copy(), pre_covariance=self.cov.cpu().numpy().copy(), motion_token=np.asarray(self.motion_tokens[-1], np.float32))
        if self.net.orientation_basis:
            forecast['headings'] = self.mean[:, 4].cpu().numpy().copy()
        if self.net.mode_evidence != 'local':
            transfer_offset = region['entry_offset'] if self.net.mode_evidence == 'frozen' else delta
            forecast.update(transfer_mean=region['transfer_mean'].cpu().numpy().copy(), transfer_covariance=region['transfer_covariance'].cpu().numpy().copy(), transfer_offset=transfer_offset.cpu().numpy().copy())
        if self.net.initial_posterior != 'none':
            outside_information, outside_vector = self._device_region_factors()
            forecast.update(prefix_region_information=outside_information.cpu().numpy().copy(), prefix_region_vector=outside_vector.cpu().numpy().copy())
        if self.net.region_reliability != 'none':
            information, vector, probability = self._closed_region_bank()
            forecast.update(prefix_closed_information=information.cpu().numpy().copy(), prefix_closed_vector=vector.cpu().numpy().copy(), prefix_closed_probability=probability.cpu().numpy().copy(), prefix_retired_information=self.retired_information.cpu().numpy().copy(), prefix_retired_vector=self.retired_vector.cpu().numpy().copy())
        if self.net.aligned_motion:
            forecast['cs_age'] = np.asarray([1000.0 if not seq else self.timestamp / 1000000000.0 - seq[-1][0] for seq in self.cs_history.values()], np.float32)
        self.forecasts.append(forecast)
        if self.net.aligned_motion:
            forecast['motion_history'] = self.current_motion_history.copy()
        if self.net.temporal_contrast:
            forecast['causal_motion_path'] = self.motion_total.cpu().numpy().copy()
        position_covariance = stabilize(self.cov[:, :2, :2])
        points, prior, g = position_support(self.mean[:, :2], position_covariance, self.grid_xy, self.position_rule, self.net.integration_grid.log_area if self.cell_rule else None)
        n = points.shape[1]
        context = encoded['motion'][0].expand(self.k, n, 32)
        global_field = self.net.field(self.grid_xy, encoded['motion'][0].expand(g, 32), self.grid_fields)
        local_field = self.net.field(points[:, g:], context[:, g:])
        field = {key: torch.cat((value[None].expand(self.k, *value.shape), local_field[key]), 1) for key, value in global_field.items()}
        field = self.net.field.condition_heading(field, points, self.mean[:, 4, None])
        field = calibrated_field(field, mu[:, None])
        basis = field['basis']
        predicted = field['mean'] + (basis @ mu[:, None, :, None]).squeeze(-1) + delta[:, None]
        noise = field['map_variance'] + encoded['sigma'][0].square()
        covariance = basis @ cov[:, None] @ basis.transpose(-1, -2) + torch.diag_embed(noise)
        m = mask.float()
        covariance = covariance * m[:, None] * m[None, :] + torch.diag(1 - m)
        loglike, radio_chol = gaussian_log_prob((y - predicted) * m, stabilize(covariance), return_factor=True)
        loglike = loglike + 0.5 * (~mask).sum() * math.log(2 * math.pi)
        device_variance = (basis @ cov[:, None] @ basis.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1)
        temporal = None
        if self.net.temporal_contrast:
            if getattr(self.net, 'direction_temporal_head', None):
                from temporal_radio import features
                temporal = tuple((value[0] for value in features(points[None], {key: value[None] for key, value in field.items()}, (predicted - field['mean'])[None], y[None], mask[None], temporal_context)))
            else:
                from base_temporal_contrast import features
                temporal = tuple((value[0] for value in features(field['gradient'][None], y[None], mask[None], delta[None], temporal_context, self.net.temporal_contrast)))
        potential = self.net.localization(y.expand(self.k, n, self.a), mask.expand(self.k, n, self.a), predicted, covariance.diagonal(dim1=-2, dim2=-1), field, device_variance, self.net.field.ap_embedding.weight, encoded['spatial_radio'][0].expand(self.k, n, self.a, 24), encoded['spatial_motion'][0].expand(self.k, n, 32), covariance, radio_chol, temporal=temporal)
        score = loglike + potential
        head_off_score = loglike
        nominal_responsibility = torch.ones_like(score)
        if self.net.robust_radio:
            if self.net.mode_spatial_evidence:
                score, nominal_responsibility, head_off_score = robust_radio_score(score, y, mask, predicted, covariance, encoded['outlier'][0], head_potential=potential)
            else:
                score, nominal_responsibility = robust_radio_score(score, y, mask, predicted, covariance, encoded['outlier'][0])
        if self.filter_cfg.get('export_trajectory_radio_factors', False):
            forecast['trajectory_radio_loglike'] = score[:, :g].cpu().numpy().copy()
        spatial_log_ratio = None
        if self.net.mode_spatial_evidence:
            spatial_log_ratio = torch.logsumexp(prior + score, -1) - torch.logsumexp(prior + head_off_score, -1)
            forecast['mode_spatial_log_ratio'] = spatial_log_ratio.cpu().numpy().copy()
        global_feedback = self.position_rule == 'hermite' or self.net.aligned_motion
        radio_points = points[:, :g] if global_feedback else points
        radio_score = score[:, :g] if global_feedback else score
        if self.cell_rule:
            radio_score = radio_score + self.net.integration_grid.log_area
        wifi_probability = radio_score.softmax(-1)
        self.last_wifi_distribution = dict(points=radio_points.clone(), probability=wifi_probability, time=self.timestamp / 1000000000.0)
        prior_position = position_covariance
        precision = torch.linalg.inv(prior_position)
        difference = points - self.mean[:, None, :2]
        density = -0.5 * torch.einsum('knd,kde,kne->kn', difference, precision, difference)
        posterior = (prior + score).softmax(-1)
        full_posterior = posterior
        spatial_projection = None
        if self.filter_cfg.get('spatial_basin_policy'):
            from base_mode_projection import basin_probability
            posterior, spatial_projection = basin_probability(points, score, self.mean[:, :2], prior_position, full_posterior, self.filter_cfg['spatial_basin_policy'], self.net.integration_grid.description()['source_cell_size_m'])
        evidence = torch.logsumexp(prior + score, -1)
        nominal_mass = (posterior * nominal_responsibility).sum(-1)
        gain = getattr(self.net.localization, 'last_gain', None)
        self.radio_diagnostics.append(dict(observation_id=event['observation_id'], sigma_db=encoded['sigma'][0].cpu().tolist(), outlier_probability=encoded['outlier'][0].cpu().tolist(), available_AP=mask.cpu().tolist(), short_residual_abs_db=float(delta.abs().mean()), spatial_precision_gain=None if gain is None else (posterior[..., None] * gain).sum(1).cpu().tolist(), nominal_responsibility=nominal_mass.cpu().tolist()))
        if spatial_log_ratio is not None:
            self.radio_diagnostics[-1]['mode_spatial_log_ratio'] = spatial_log_ratio.cpu().tolist()
        coupling = self.cov[:, 2:, :2] @ precision
        conditional_mean = self.mean[:, None, 2:] + torch.einsum('kud,knd->knu', coupling, difference)
        conditional_cov = stabilize(self.cov[:, 2:, 2:] - coupling @ self.cov[:, :2, 2:])
        hmean = field['mean'] + (basis @ conditional_mean[..., self.eta_u, None]).squeeze(-1) + delta[:, None]
        update_noise = noise / nominal_responsibility.clamp_min(0.001)[..., None] if self.net.robust_radio else noise
        s = basis @ conditional_cov[:, None, self.eta_u, self.eta_u] @ basis.transpose(-1, -2) + torch.diag_embed(update_noise)
        s = s * m[:, None] * m[None, :] + torch.diag(1 - m)
        cross = conditional_cov[:, None, :, self.eta_u] @ basis.transpose(-1, -2) * m
        gain = torch.linalg.solve(stabilize(s), cross.transpose(-1, -2)).transpose(-1, -2)
        conditional_mean = conditional_mean + (gain @ ((y - hmean) * m)[..., None]).squeeze(-1)
        conditional_cov = stabilize(conditional_cov[:, None] - gain @ cross.transpose(-1, -2))
        components = torch.cat((points, conditional_mean), -1)
        point_probability = posterior
        if self.filter_cfg.get('spatial_point_rule') == 'geometric_median':
            location = (posterior[..., None] * points).sum(1)
            for _ in range(32):
                distance = ((points - location[:, None]).square().sum(-1) + 0.02 ** 2).sqrt()
                risk_weight = posterior / distance
                risk_weight = risk_weight / risk_weight.sum(-1, keepdim=True).clamp_min(1e-20)
                updated = (risk_weight[..., None] * points).sum(1)
                change = (updated - location).norm(dim=-1).max()
                location = updated
                if float(change) < 0.0001:
                    break
            informative = score.max(-1).values - score.min(-1).values > 1e-05
            point_probability = torch.where(informative[:, None], risk_weight, posterior)
        mean = (point_probability[..., None] * components).sum(1)
        difference_full = components - mean[:, None]
        covariance_full = torch.einsum('kn,knd,kne->kde', full_posterior, difference_full, difference_full)
        covariance_full[:, 2:, 2:] += (full_posterior[..., None, None] * conditional_cov).sum(1)
        if spatial_projection is not None:
            self.radio_diagnostics[-1].update(spatial_peak_count=spatial_projection['peak_count'].cpu().tolist(), selected_spatial_basin_probability=spatial_projection['selected_basin_probability'].cpu().tolist(), spatial_projection_shift_m=((full_posterior[..., None] * components[:, :, :2]).sum(1) - mean[:, :2]).norm(dim=-1).cpu().tolist())
        if self.position_rule == 'mixed':
            cell_variance = (posterior[:, :g] * self.net.integration_grid.variance).sum(-1) if self.cell_rule else 0.6 ** 2 / 12 * full_posterior[:, :g].sum(-1)
            covariance_full[:, :2, :2] += torch.eye(2, device=self.device) * cell_variance[:, None, None]
        if self.filter_cfg.get('center_weak_prior_moment', False):
            prior_bias = (prior.softmax(-1)[..., None] * points).sum(1) - self.mean[:, :2]
            root = torch.linalg.cholesky(prior_position)
            whitened = torch.linalg.solve_triangular(root, covariance_full[:, :2, :2], upper=False)
            whitened = torch.linalg.solve_triangular(root, whitened.transpose(-1, -2), upper=False).transpose(-1, -2)
            values, vectors = torch.linalg.eigh(0.5 * (whitened + whitened.transpose(-1, -2)))
            sensitivity = vectors * values.clamp(0.0, 1.0)[:, None, :] @ vectors.transpose(-1, -2)
            white_bias = torch.linalg.solve_triangular(root, prior_bias[..., None], upper=False)
            correction = (root @ sensitivity @ white_bias).squeeze(-1)
            mean = mean - torch.cat((correction, (coupling @ correction[..., None]).squeeze(-1)), -1)
            self.radio_diagnostics[-1].update(numerical_prior_bias_m=prior_bias.cpu().tolist(), numerical_prior_correction_m=correction.cpu().tolist())
        if self.map_refine:
            best = (score + density).argmax(-1)
            start = points[torch.arange(self.k, device=self.device), best]
            refined = self._refine_position(start, self.mean[:, :2], prior_position, encoded, mu, cov, delta, y, mask, event['observation_id'])
            conditional = self.mean[:, 2:] + torch.einsum('kud,kd->ku', coupling, refined - self.mean[:, :2])
            selected_field = self.net.field(refined, encoded['motion'][0].expand(self.k, 32), heading=self.mean[:, 4])
            selected_field = calibrated_field(selected_field, mu)
            selected_basis = selected_field['basis']
            selected_noise = selected_field['map_variance'] + encoded['sigma'][0].square()
            selected_mean = selected_field['mean'] + (selected_basis @ conditional[:, self.eta_u, None]).squeeze(-1) + delta
            base_conditional = stabilize(self.cov[:, 2:, 2:] - coupling @ self.cov[:, :2, 2:])
            selected_s = selected_basis @ base_conditional[:, self.eta_u, self.eta_u] @ selected_basis.transpose(-1, -2) + torch.diag_embed(selected_noise)
            selected_s = selected_s * m[:, None] * m[None, :] + torch.diag(1 - m)
            selected_cross = base_conditional[:, :, self.eta_u] @ selected_basis.transpose(-1, -2) * m
            selected_gain = torch.linalg.solve(stabilize(selected_s), selected_cross.transpose(-1, -2)).transpose(-1, -2)
            conditional = conditional + (selected_gain @ ((y - selected_mean) * m)[..., None]).squeeze(-1)
            adopted = torch.cat((refined, conditional), -1)
            offset = mean - adopted
            covariance_full = covariance_full + offset[..., None] * offset[:, None]
            mean = adopted
        old_position = self.mean[:, :2].clone()
        old_variance = prior_position.diagonal(dim1=-2, dim2=-1).sum(-1)
        old_field = self.net.field(old_position, encoded['motion'][0].expand(self.k, 32), heading=self.mean[:, 4])
        old_field = calibrated_field(old_field, mu)
        old_uncalibrated = old_field['mean'] + (old_field['basis'] @ mu[..., None]).squeeze(-1)
        innovation = y - old_uncalibrated
        innovation_design = old_field['basis']
        if self.net.project_innovations:
            nuisance = old_field['basis'] @ cov @ old_field['basis'].transpose(-1, -2) + torch.diag_embed(old_field['map_variance'] + encoded['sigma'][0].square())
            pose = old_field['gradient'] @ prior_position @ old_field['gradient'].transpose(-1, -2)
            valid = mask.float()
            nuisance = nuisance * valid[:, None] * valid[None, :] + torch.diag(1 - valid)
            pose = pose * valid[:, None] * valid[None, :]
            innovation = (nuisance @ torch.linalg.solve(stabilize(nuisance + pose), (innovation * valid)[..., None])).squeeze(-1) * valid
            if self.net.memory_recenter:
                innovation_design = nuisance @ torch.linalg.solve(stabilize(nuisance + pose), old_field['basis'] * valid[..., None]) * valid[..., None]
        self.innovations.append(innovation.detach())
        if self.net.memory_recenter:
            self.innovation_designs.append(innovation_design.detach())
            self.innovation_device_means.append(mu.detach())
            forecast['causal_device_design'] = innovation_design.cpu().numpy().copy()
        forecast['causal_innovation'] = innovation.cpu().numpy().copy()
        self.predicted_poses.append(old_position)
        self.pose_variances.append(old_variance)
        self.mean = mean
        self.cov = stabilize(covariance_full)
        if self.net.mode_evidence != 'local':
            from base_transfer_evidence import predictive_transfer
            evidence, next_mean, next_cov = predictive_transfer(y, mask, old_field, region['transfer_mean'], region['transfer_covariance'], transfer_offset, encoded['sigma'][0], encoded['outlier'][0], prior_position, update=True, robust=self.net.robust_transfer)
            region['transfer_mean'] = next_mean
            region['transfer_covariance'] = next_cov
            forecast['transfer_log_evidence'] = evidence.cpu().numpy().copy()
        if spatial_log_ratio is not None:
            evidence = evidence + spatial_log_ratio
        self.log_weight = (self.log_weight + evidence).log_softmax(-1)
        expected_basis = (posterior[..., None, None] * basis).sum(1)
        expected_source = (posterior[..., None] * field['mean']).sum(1)
        gradient = (posterior[..., None, None] * field['gradient']).sum(1)
        if self.map_refine:
            expected_basis = selected_basis
            expected_source = selected_field['mean']
            gradient = selected_field['gradient']
            noise = selected_noise[:, None].expand(self.k, n, self.a)
        pose_noise = torch.einsum('kad,kde,kae->ka', gradient, self.cov[:, :2, :2], gradient).clamp_min(0.0)
        effective_noise = (posterior[..., None] * noise).sum(1) + pose_noise
        effective_noise = effective_noise / nominal_mass.clamp_min(0.001)[:, None] if self.net.robust_radio else effective_noise / (1 - encoded['outlier'][0]).clamp_min(0.05)
        inv = mask.float() / effective_noise.clamp_min(0.25)
        information = torch.einsum('kad,ka,kae->kde', expected_basis, inv, expected_basis)
        vector = torch.einsum('kad,ka,ka->kd', expected_basis, inv, y - expected_source - delta)
        if self.net.full_pose_covariance:
            support_noise = torch.diag_embed(old_field['map_variance'] + encoded['sigma'][0].square()) + old_field['gradient'] @ prior_position @ old_field['gradient'].transpose(-1, -2)
            if self.net.robust_radio:
                support_noise = support_noise / nominal_mass.clamp_min(0.001)[:, None, None]
            else:
                scale = (1 - encoded['outlier'][0]).clamp_min(0.05).rsqrt()
                support_noise = support_noise * scale[:, None] * scale[None, :]
            information, vector, _, _, _ = self.net.calibration.block_information(old_field['basis'][:, None], (y - old_field['mean'] - delta)[:, None], support_noise[:, None], mask.expand(self.k, 1, self.a))
        initial_window_active = self.net.initial_recognition != 'first' and len(self.radio) < 5
        if initial_window_active:
            from base_initial_posterior import window_posterior
            self.initial_window.append(dict(rssi=(y - delta).clone(), mask=mask.expand(self.k, self.a).clone(), mean=old_field['mean'].clone(), basis=old_field['basis'].clone(), gradient=old_field['gradient'].clone(), noise=support_noise.clone()))
            window = {key: torch.stack([row[key] for row in self.initial_window], 1) for key in self.initial_window[0]}
            self.initial_information, self.initial_vector, self.initial_device_posterior = window_posterior(self.net.calibration, window['rssi'], window['mask'], window, window['noise'], self.net.initial_recognition, self.net.field.ap_embedding.weight)
            self.initial_ids.append(event['observation_id'])
        elif self.net.initial_context_scans > 1 and len(self.radio) < self.net.initial_context_scans:
            initial_pose_noise = torch.einsum('kad,kde,kae->ka', old_field['gradient'], prior_position, old_field['gradient']).clamp_min(0.0)
            initial_noise = old_field['map_variance'] + encoded['sigma'][0].square() + initial_pose_noise
            initial_noise = initial_noise / nominal_mass.clamp_min(0.001)[:, None] if self.net.robust_radio else initial_noise / (1 - encoded['outlier'][0]).clamp_min(0.05)
            weight = mask.float() / initial_noise.clamp_min(0.25)
            self.initial_mass += weight
            self.initial_basis_sum += weight[..., None] * old_field['basis']
            self.initial_offset_sum += weight * (y - old_field['mean'] - delta)
            self.initial_noise_sum += weight * initial_noise
            mass = self.initial_mass.clamp_min(1e-08)
            average_basis = self.initial_basis_sum / mass[..., None]
            average_offset = self.initial_offset_sum / mass
            average_noise = self.initial_noise_sum / mass
            inverse = (self.initial_mass > 0).float() / average_noise.clamp_min(0.25)
            self.initial_information = torch.einsum('kad,ka,kae->kde', average_basis, inverse, average_basis)
            self.initial_vector = torch.einsum('kad,ka,ka->kd', average_basis, inverse, average_offset)
            self.initial_ids.append(event['observation_id'])
        elif not self.radio:
            self.initial_information = information
            self.initial_vector = vector
            self.initial_ids.append(event['observation_id'])
        else:
            region = self.regions[-1]
            region['information'] += information
            region['vector'] += vector
            region['observation_ids'].append(event['observation_id'])
        if self.radio:
            self.regions[-1]['count'] += 1
        if initial_window_active or (not self.radio and self.net.initial_posterior != 'none'):
            if not initial_window_active:
                base_mean, base_cov = self.net.calibration.posterior(self.initial_information, self.initial_vector)
                self.initial_device_posterior = self.net.calibration.initial_encoder(y, mask, old_field, support_noise.diagonal(dim1=-2, dim2=-1), base_mean, base_cov, self.net.field.ap_embedding.weight)
            if self.net.mode_evidence != 'local':
                region['transfer_mean'] = self.initial_device_posterior[0].clone()
                region['transfer_covariance'] = self.initial_device_posterior[1].clone()
        if self.net.initial_posterior != 'none':
            all_info = self.retired_information.clone()
            all_vector = self.retired_vector.clone()
        else:
            all_info = self.initial_information + self.retired_information
            all_vector = self.initial_vector + self.retired_vector
        if self.net.region_reliability != 'none':
            from base_region_reliability import shared_probability
            all_info, all_vector = self._device_region_factors()
            reference_mean, reference_covariance = self.net.calibration.posterior(all_info, all_vector, *self.initial_device_posterior)
            current = self.regions[-1]
            probability, bayes_factor = shared_probability(self.net.calibration, current['information'], current['vector'], reference_mean, reference_covariance, *self.initial_device_posterior)
            current['original_probability'] = probability
            all_info = all_info + probability[:, None, None] * current['information']
            all_vector = all_vector + probability[:, None] * current['vector']
            self.radio_diagnostics[-1].update(region_shared_probability=probability.tolist(), region_log_bayes_factor=bayes_factor.tolist())
        else:
            for region in self.regions:
                all_info = all_info + region['information']
                all_vector = all_vector + region['vector']
        new_mu, new_cov = self.net.calibration.posterior(all_info, all_vector, *(self.initial_device_posterior if self.net.initial_posterior != 'none' else (None, None)))
        if self.temporal:
            self.mean, self.cov = replace_device(self.mean, self.cov, new_mu, new_cov)
        else:
            old_cov = stabilize(self.cov[:, self.eta, self.eta])
            old_chol = torch.linalg.cholesky(old_cov)
            new_chol = torch.linalg.cholesky(stabilize(new_cov))
            transport = torch.linalg.solve_triangular(old_chol.transpose(-1, -2), new_chol.transpose(-1, -2), upper=True)
            cross = self.cov[:, :, self.eta] @ transport
            self.mean[:, self.eta] = new_mu
            self.cov[:, :, self.eta] = cross
            self.cov[:, self.eta, :] = cross.transpose(-1, -2)
            self.cov[:, self.eta, self.eta] = new_cov
            self.cov = stabilize(self.cov)
        self.rssi_history.append(y.cpu().numpy())
        self.mask_history.append(mask.cpu().numpy())
        self.times.append(self.timestamp / 1000000000.0)
        if self.net.temporal_contrast:
            self.motion_paths.append(self.motion_total.clone())
            self.delta_history.append(delta.clone())
        self.wifi_axes.append(np.asarray(self.motion_tokens[-1])[[0, 2]])
        tokens = radio_tokens(np.asarray(self.rssi_history), np.asarray(self.mask_history), np.asarray(self.times))
        self.radio.append(tokens[-1])
        weight = self.log_weight.softmax(-1)
        position = (weight[:, None] * self.mean[:, :2]).sum(0)
        region = self.regions[-1]
        spatial_distance = float(np.linalg.norm(position.cpu().numpy() - region['origin']))
        if region['count'] >= 4 and spatial_distance >= 1.5 or (region['count'] >= 12 and spatial_distance >= 0.6):
            self.regions.append(self._region(position.cpu().numpy()))
            if len(self.regions) > 4:
                retired = self.regions.pop(0)
                probability = retired['original_probability'] if self.net.region_reliability != 'none' else torch.ones(self.k, device=self.device)
                self.retired_information += probability[:, None, None] * retired['information']
                self.retired_vector += probability[:, None] * retired['vector']
        output = dict(observation_id=event['observation_id'], timestamp_ns=self.timestamp, xy_m=position.cpu().tolist(), mode_xy_m=self.mean[:, :2].cpu().tolist(), mode_weights=weight.cpu().tolist(), region_count=len(self.regions), known_initial_position_only=True)
        if self.filter_cfg.get('position_readout') == 'posterior_Euclidean_risk':
            point = points.reshape(-1, 2)
            probability = (weight[:, None] * full_posterior).reshape(-1)
            location = position
            for _ in range(32):
                distance = ((point - location).square().sum(-1) + 0.02 ** 2).sqrt()
                action_weight = probability / distance
                action_weight = action_weight / action_weight.sum().clamp_min(1e-20)
                updated = (action_weight[:, None] * point).sum(0)
                change = (updated - location).norm()
                location = updated
                if float(change) < 0.0001:
                    break
            output.update(xy_m=location.cpu().tolist(), posterior_mean_xy_m=position.cpu().tolist(), point_estimate_rule='posterior Euclidean-risk action; no state/calibration replacement')
        self.outputs.append(output)
        return output

    @torch.no_grad()
    def consume(self, event):
        oid = event['observation_id']
        if oid in self.ledger:
            return
        if any((k in event.get('payload', {}) for k in ('gt_xy_m', 'true_position', 'true_velocity'))):
            raise ValueError('Truth is not an inference input')
        self._advance(event['monotonic_timestamp_ns'])
        if event['kind'] == 'imu':
            from support_motion import rotation_matrix
            rotation = rotation_matrix(event['payload']['game_rotation_quat'])
            if self.first_rotation is None:
                self.first_rotation = rotation.copy()
            linear = np.asarray(event['payload']['acc_xyz']) - 9.80665 * rotation[2, :]
            self.acceleration = np.clip((self.first_rotation.T @ rotation @ linear)[:2], -5, 5)
            self.motion_observer.consume(SimpleNamespace(**event))
            self.motion.observe(event['payload'])
        else:
            self._motion_token(event)
            if event['kind'] == 'cs':
                self._consume_cs(event)
            elif event['kind'] == 'wifi':
                output = self._consume_wifi(event)
                self.ledger.add(oid)
                return output
        self.ledger.add(oid)
