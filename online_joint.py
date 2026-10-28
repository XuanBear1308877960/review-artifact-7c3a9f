import copy
import math
import os
import time
import numpy as np
import torch
from torch import nn
from online_m2 import RegionalCalibration

def query(assets, xy, motion, shifts=None, weights=None):
    xy = np.asarray(xy)
    shape = xy.shape[:-1]
    p = xy.reshape(-1, 2)
    context = np.broadcast_to(np.asarray(motion), shape + (32,)).reshape(-1, 32)
    points = np.repeat(p, 4, 0)
    contexts = np.repeat(context, 4, 0)
    ap = np.tile(np.arange(4), len(p))
    if shifts is not None:
        points = points[:, None] + np.asarray(shifts)[ap]
        contexts = np.repeat(contexts, 2, axis=0)
        indices = np.repeat(ap, 2)
        values = assets['radio'].ap_query(points.reshape(-1, 2), contexts, indices, True)
        reduced = []
        for v in values:
            v = np.asarray(v).reshape(len(ap), 2, *np.asarray(v).shape[1:])
            w = np.asarray(weights)[ap].reshape(len(ap), 2, *[1] * (v.ndim - 2))
            reduced.append((v * w).sum(1))
        values = reduced
    else:
        values = assets['radio'].ap_query(points, contexts, ap, True)
    return tuple((np.asarray(v).reshape(*shape, *tail) for v, tail in zip(values, [(4,), (4, 6), (4, 2), (4, 6, 2)])))

class Field(nn.Module):

    def __init__(self, base, assets):
        super().__init__()
        self.base = base
        self.assets = assets
        self.shifts = None
        self.weights = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('base'), name)

    def forward(self, points, context, fields=None, heading=None):
        result = self.base(points, context, fields, heading)
        motion = context.reshape(-1, 32)[0].detach().numpy()
        m, B, g, dB = query(self.assets, points.detach().numpy(), motion, self.shifts, self.weights)
        result.update(mean=torch.from_numpy(m).float(), basis=torch.from_numpy(B).float(), gradient=torch.from_numpy(g).float(), basis_gradient=torch.from_numpy(dB).float(), _online_xy=points)
        return result

class Readout(nn.Module):

    def __init__(self, model, grid):
        super().__init__()
        self.model = model
        self.post = None
        self.noise = None
        self.conditional = None
        self.register_buffer('grid', torch.tensor(grid, dtype=torch.float32))

    def forward(self, y, mask, predicted, variance, field, device_variance, embedding, *args, **kwargs):
        xy = field['_online_xy']
        shape = xy.shape[:-1]
        distance = (xy[..., None, :] - self.grid).square().sum(-1)
        if xy.ndim != 3:
            raise ValueError('Expected mode x spatial-point x xy')
        ix = distance.argmin(1)
        if self.post is not None:
            from base_model import gaussian_log_prob, stabilize
            mu, C, lw = self.post
            B = field['basis']
            m = field['mean']
            if self.conditional is not None:
                offset, offset_B, noise = self.conditional
                m = m + offset[:, None]
                B = B - offset_B[:, None]
            else:
                noise = self.noise[None].expand(xy.shape[0], -1)
            mapped = torch.einsum('knad,cde,knae->knac', B, C, B)
            prediction = m[..., None] + torch.einsum('knad,cd->knac', B, mu)
            v = (noise[:, None, :, None] + mapped).clamp_min(0.25)
            residual = y[..., None] - prediction
            ap = -0.5 * (residual.square() / v + v.log() + math.log(2 * math.pi))
            gaussian = torch.logsumexp((ap * mask[..., None]).sum(-2) + lw, -1)
            w = lw.softmax(-1)
            average = (prediction * w).sum(-1)
            total = (v * w).sum(-1) + ((prediction - average[..., None]).square() * w).sum(-1)
            gradient = field.get('source_gradient', field['gradient']) + torch.einsum('knadz,cd,c->knaz', field['basis_gradient'], mu, w)
            scalar = torch.stack(((y - average) / total.sqrt(), total.log() / 5, field['mean'] / 50, gradient[..., 0] / 10, gradient[..., 1] / 10, (mapped * w).sum(-1) / 100, field['visibility'], torch.ones_like(m)), -1)
            feature = torch.cat((scalar, embedding.expand(*predicted.shape, 8)), -1)
            selected = feature.gather(1, ix[..., None, None].expand(-1, -1, feature.shape[-2], 16))
            correction = self.model.positioning(selected, mask[:, 0])
            kernel = -distance / (2 * 0.35 ** 2)
            interpolated = torch.logsumexp(kernel + correction[:, None], -1) - torch.logsumexp(kernel, -1)
            covariance = args[2]
            observed = mask[0, 0].float()
            old = gaussian_log_prob((y - predicted) * observed, stabilize(covariance))
            old = old + 0.5 * (~mask[0, 0]).sum() * math.log(2 * math.pi)
            return 2 * (gaussian + interpolated) - old
        gather = lambda value: value.gather(1, ix[..., None].expand(-1, -1, value.shape[-1]))
        v = variance.clamp_min(0.25)
        scalar = torch.stack(((y - predicted) / v.sqrt(), v.log() / 5, field['mean'] / 50, field['gradient'][..., 0] / 10, field['gradient'][..., 1] / 10, device_variance / 100, field['visibility'], torch.ones_like(predicted)), -1)
        feature = torch.cat((scalar, embedding.expand(*predicted.shape, 8)), -1)
        selected = feature.gather(1, ix[..., None, None].expand(-1, -1, feature.shape[-2], 16))
        correction = self.model.positioning(selected, mask[:, 0])
        kernel = -distance / (2 * 0.35 ** 2)
        return torch.logsumexp(kernel + correction[:, None], -1) - torch.logsumexp(kernel, -1)

def create_engine(assets, initial_xy, initial_ns):
    from kinematics import RadioDirectionEngine
    from base_cs_temporal import replace_device

    class FullCS(RadioDirectionEngine):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            from online_cs import ForwardCS
            self.forward_cs = ForwardCS(assets['prior'])
            from online_motion import VelocityStates
            self.velocity_states = VelocityStates(assets)

        def consume(self, event):
            out = super().consume(event)
            if os.environ.get('YOMO_MOBILE_ONLINE_VELOCITY', '0') == '1' and event['kind'] == 'imu':
                self.velocity_states.update(self, event)
            return out

        def _consume_cs(self, event):
            if os.environ.get('YOMO_MOBILE_ONLINE_RAW_CS', '0') == '1':
                return self.forward_cs.update(self, event)
            return super()._consume_cs(event)

    class Geometry(FullCS):

        def _consume_wifi(self, event):
            if len(self.outputs) < 5:
                return super()._consume_wifi(event)
            out = dict(observation_id=event['observation_id'], timestamp_ns=self.timestamp, xy_m=(self.log_weight.softmax(-1)[:, None] * self.mean[:, :2]).sum(0).tolist())
            self.outputs.append(out)
            return out

    class Online(FullCS):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calibrator = None
            self.steps = 0
            self.cavities = []
            self.is_cavity = False
            self.online_diagnostics = []
            self.active_posterior = None
            from online_radio_noise import RadioNoise
            self.radio_noise = RadioNoise(assets['prior']['temporal_RSSI_parameters'])
            from online_lag import SupportLag
            self.support_lag = SupportLag(12)
            self.lag_prior = None
            self.position_trace = [(self.timestamp / 1000000000.0, self.mean[:, :2].numpy().copy())]
            self.start_ns = self.timestamp
            self.base_variance = assets['base_variance']
            self.full_net = copy.copy(self.net)
            self.full_net._modules = self.net._modules.copy()
            self.full_net.field = Field(self.net.field, assets)
            self.full_net.localization = Readout(assets['model'], assets['grid'])
            if os.environ.get('YOMO_MOBILE_ONLINE_CLEAN_M2', '0') == '1':
                self.full_net.mode_evidence = 'local'
                self.full_net.mode_spatial_evidence = False
                self.full_net.region_reliability = 'none'
                self.full_net.robust_radio = False

        def fork_runtime(self):
            self.full_net = copy.copy(self.full_net)
            self.full_net._modules = self.full_net._modules.copy()
            self.full_net.field = Field(self.full_net.field.base, assets)
            self.full_net.localization = Readout(assets['model'], assets['grid'])
            self.net = self.full_net

        def consume(self, event):
            out = super().consume(event)
            self.position_trace.append((self.timestamp / 1000000000.0, self.mean[:, :2].numpy().copy()))
            self.position_trace = self.position_trace[-128:]
            return out

        def _device_prediction(self):
            if self.active_posterior is None:
                return super()._device_prediction()
            return self.active_posterior

        def update_position(self, event):
            return super()._consume_wifi(event)

        def _advance(self, ns):
            dt = (int(ns) - self.timestamp) / 1000000000.0
            super()._advance(ns)
            if hasattr(self, 'support_lag'):
                self.support_lag.predict(dt)
                self.lag_prior = (self.mean[:, :4].numpy().copy(), self.cov[:, :4, :4].numpy().copy())

        def update_support_geometry(self):
            if os.environ.get('YOMO_MOBILE_ONLINE_SUPPORT_LAG', '0') != '1' or not self.is_cavity or self.lag_prior is None:
                return
            self.support_lag.observe(*self.lag_prior, self.mean[:, :4].numpy(), self.cov[:, :4, :4].numpy())
            active = {id(row) for row in self.calibrator.rows}
            rows = [r for r in self.support_lag.rows if id(r['factor']) in active]
            if not rows:
                return
            xy = np.stack([r['xy'] for r in rows])
            motion = np.stack([np.broadcast_to(r['motion'], (8, 32)) for r in rows])
            means, bases, gradients, _ = query(assets, xy, motion)
            for i, r in enumerate(rows):
                row = r['factor']
                self.calibrator._change(row, -1)
                row['mean'] = torch.tensor(means[i] + r['offset'], dtype=torch.float64)
                row['basis'] = torch.tensor(bases[i] - r['basis_offset'], dtype=torch.float64)
                pose = np.einsum('kad,kde,kae->ka', gradients[i], r['cov'], gradients[i]).clip(0)
                row['variance'] = torch.tensor((r['noise_without_pose'] + pose).clip(0.25), dtype=torch.float64)
                B = row['basis']
                z = row['y'][None] - row['mean']
                w = row['mask'][None] / row['variance']
                row.update(J=torch.einsum('ka,kad,kae->kde', w, B, B), b=torch.einsum('ka,kad,ka->kd', w, B, z), c=(w * z.square()).sum(-1), ld=(row['mask'][None] * row['variance'].log()).sum(-1))
                self.calibrator._change(row, 1)

        def _consume_cs(self, event):
            result = super()._consume_cs(event)
            if self.calibrator is not None:
                self.update_support_geometry()
            return result

        def _consume_wifi(self, event):
            if self.steps < 5:
                out = super()._consume_wifi(event)
                self.steps += 1
                if self.steps == 5:
                    mu, P = self.geometry.initial_device_posterior
                    self.calibrator = RegionalCalibration(assets['model'], (mu, P, self.geometry.log_weight), self.net.field.ap_embedding.weight, 48)
                    self.net = self.full_net
                return out
            observations = event['payload']['ap_observations']
            if not observations:
                return None
            metadata = event['payload'].get('RSSI_contributors')
            if metadata is not None:
                observations = [ap for ap in observations if self.lookup.get(ap['ap_id_hash']) is not None and (not metadata['initial_reused'][self.lookup[ap['ap_id_hash']]]) and (np.sum(metadata['weights'][self.lookup[ap['ap_id_hash']]]) > 0)]
                event = dict(event, payload=dict(event['payload'], ap_observations=observations))
                if not observations:
                    self.steps += 1
                    weight = self.log_weight.softmax(-1)
                    out = dict(observation_id=event['observation_id'], timestamp_ns=self.timestamp, xy_m=(weight[:, None] * self.mean[:, :2]).sum(0).tolist(), no_new_radio_factor=True)
                    self.outputs.append(out)
                    self.online_diagnostics.append(dict(observation_id=event['observation_id'], M2_predict_ms=0.0, M2_increment_ms=0.0, M3_ms=0.0, context_rows=len(self.calibrator.rows), context_updates=self.calibrator.updates, M3_nodes=6, pose_modes=8, solver_iterations=0, no_new_radio_factor=True))
                    return out
            if metadata is not None and os.environ.get('YOMO_MOBILE_ONLINE_CONTRIBUTOR_GEOMETRY', '0') == '1':
                history = self.position_trace
                times = np.array([r[0] for r in history])
                mode = int(self.log_weight.argmax())
                path = np.array([r[1][mode] for r in history])
                at = np.array(metadata['times_s']) + self.start_ns / 1000000000.0
                shifts = np.stack([np.interp(at, times, path[:, j]) for j in range(2)], -1) - self.mean[mode, :2].numpy()
                weights = np.array(metadata['weights'], float)
                missing = weights.sum(-1) <= 0
                weights[missing] = [1.0, 0.0]
                shifts[missing] = 0.0
                self.net.field.shifts = shifts
                self.net.field.weights = weights
            y = np.zeros(4)
            mask = np.zeros(4, bool)
            budget = np.zeros(4)
            for ap in observations:
                a = self.lookup.get(ap['ap_id_hash'])
                if a is not None:
                    y[a] = ap['rssi_dbm']
                    mask[a] = True
                    budget[a] = 1 / max(1, int(ap.get('rssi_preprocessing', {}).get('history_n', 1)))
            origin = (self.geometry.log_weight.softmax(-1)[:, None] * self.geometry.mean[:, :2]).sum(0).numpy()
            cell = np.floor(origin / 2).astype(int)
            region = int((int(cell[0]) * 73856093 ^ int(cell[1]) * 19349663) % 4)
            calibration = self.calibrator if self.is_cavity else self.cavities[region].calibrator
            context = self if self.is_cavity else self.cavities[region]
            start = time.perf_counter_ns()
            post, diag = calibration.cavity(region, context.log_weight)
            mu, P, lw = post
            w = lw.softmax(-1)
            m = (w[:, None] * mu).sum(0)
            d = mu - m
            C = (w[:, None, None] * (P + d[:, :, None] * d[:, None, :])).sum(0)
            regions = [j for j in range(4) if j != region and calibration.region_count[j] > 0]
            if regions:
                pm, pC, pl = calibration.posterior(regions)
                pw = pl.softmax(-1)
                conditional_mean = (pw[..., None] * pm).sum(1)
                displacement = pm - conditional_mean[:, None]
                conditional_cov = (pw[..., None, None] * (pC + displacement[..., None] * displacement[..., None, :])).sum(1)
                self.active_posterior = (conditional_mean.float(), conditional_cov.float())
            else:
                self.active_posterior = tuple((v.float().clone() for v in calibration.prior[:2]))
            self.net.localization.post = post
            self.net.localization.noise = torch.tensor(self.base_variance, dtype=torch.float32)
            if calibration.last_sigma is not None:
                self.net.localization.noise += ((calibration.last_sigma.square() + 64 * calibration.last_pi) * w[:, None]).sum(0).float()
            m2_ms = (time.perf_counter_ns() - start) / 1000000.0
            self.mean, self.cov = replace_device(self.mean, self.cov, *self.active_posterior)
            positions = self.mean[:, :2].numpy().copy()
            pose_cov = self.cov[:, :2, :2].numpy().copy()
            encoded = self._encoded(torch.tensor(mask))
            motion = encoded['motion'][0].numpy()
            gm, B, gradient, _ = query(assets, positions, motion, self.net.field.shifts, self.net.field.weights)
            projected = np.einsum('kad,kde,kae->ka', gradient, pose_cov, gradient).clip(0)
            predicted = gm + np.einsum('kad,d->ka', B, m.numpy())
            dr = np.zeros((8, 4)) if not hasattr(self, 'last_y') else np.broadcast_to(y - self.last_y, (8, 4))
            dp = np.zeros_like(predicted) if not hasattr(self, 'last_prediction') else predicted - self.last_prediction
            dt = 0.0 if not hasattr(self, 'last_ns') else (self.timestamp - self.last_ns) / 1000000000.0
            band = np.unique(assets['radio'].band, axis=0, return_inverse=True)[1]
            encoder = np.stack((np.broadcast_to(y / 50, (8, 4)), predicted / 50, (y - predicted) / 10, dr / 10, dp / 10, np.broadcast_to(np.log1p(self.base_variance) / 5, (8, 4)), gradient[..., 0] / 10, gradient[..., 1] / 10, projected / 100, np.broadcast_to(band, (8, 4)), np.full((8, 4), dt / 10), np.broadcast_to(budget, (8, 4))), -1)
            factor_mean = gm
            factor_basis = B
            factor_variance = self.base_variance[None] + projected
            if os.environ.get('YOMO_MOBILE_ONLINE_RADIO_NOISE', '0') == '1':
                quality = (self.net.localization.noise - torch.tensor(self.base_variance, dtype=torch.float32)).numpy()
                offset, offset_B, noise = self.radio_noise.prepare(event, y, gm, B, projected, mask, quality)
                self.net.localization.conditional = tuple((torch.tensor(v, dtype=torch.float32) for v in (offset, offset_B, noise)))
                factor_mean = gm + offset
                factor_basis = B - offset_B
                factor_variance = (noise - quality[None]).clip(0.05) + projected
                budget = mask.astype(float)
            start = time.perf_counter_ns()
            out = self.update_position(event)
            m3_ms = (time.perf_counter_ns() - start) / 1000000.0
            if os.environ.get('YOMO_MOBILE_ONLINE_CLEAN_M2', '0') == '1':
                self.mean, self.cov = replace_device(self.mean, self.cov, *self.active_posterior)
            start = time.perf_counter_ns()
            if self.is_cavity:
                row = dict(region=region, y=torch.tensor(y), mean=torch.tensor(factor_mean), basis=torch.tensor(factor_basis), variance=torch.tensor(factor_variance), mask=torch.tensor(budget), encoder=torch.tensor(encoder), oid=event['observation_id'])
                self.calibrator.add(row)
                if os.environ.get('YOMO_MOBILE_ONLINE_SUPPORT_LAG', '0') == '1':
                    self.support_lag.append(row, np.column_stack((positions, self.lag_prior[0][:, 2:4])), self.lag_prior[1], motion, factor_mean - gm, B - factor_basis, projected)
                    self.update_support_geometry()
            increment_ms = (time.perf_counter_ns() - start) / 1000000.0
            self.last_y = y.copy()
            self.last_prediction = predicted.copy()
            self.last_ns = self.timestamp
            self.steps += 1
            self.online_diagnostics.append(dict(observation_id=event['observation_id'], M2_predict_ms=m2_ms, M2_increment_ms=increment_ms, M3_ms=m3_ms, context_rows=len(self.calibrator.rows), context_updates=self.calibrator.updates, excluded_region=region, **diag, M3_nodes=6, pose_modes=8, solver_iterations=0))
            return out
    params = np.asarray(assets['prior']['temporal_RSSI_parameters'])
    assets['base_variance'] = np.exp(2 * params[:4]) + np.exp(2 * params[4:8]) + np.exp(2 * params[9:13])
    cfg = copy.deepcopy(assets['cfg'])
    cfg['filter'].update(direction_policy='radio_frame_independent_cs', export_trajectory_radio_factors=False)
    motion = os.environ.get('YOMO_MOBILE_ONLINE_MOTION', 'frame')
    if motion in ('cv', 'speed'):
        cfg['filter']['direction_policy'] = 'radio_velocity_mixture' if motion == 'cv' else 'radio_speed_magnitude'
        assets['net'].motion_speed_head = assets['speed_head']
    if os.environ.get('YOMO_MOBILE_ONLINE_VELOCITY', '0') == '1':
        cfg['filter'].update(direction_policy='radio_velocity_mixture', acceleration_process_noise=float(np.exp(assets['prior']['marginal_force_parameters'][1])))
    args = (assets['net'], assets['aps'], cfg['anchors'], initial_xy, initial_ns, cfg['filter'], False, 1, 'mixed')
    geometry = Geometry(*args)
    main = Online(*args)
    main.geometry = geometry
    if os.environ.get('YOMO_MOBILE_ONLINE_VELOCITY', '0') == '1':
        from trajectory_axial_motion import axis_log_density
        for engine in [geometry, main]:
            value = [axis_log_density(j * math.pi / 4, assets['prior']['axial_motion_prior_parameters']) for j in range(8)]
            engine.log_weight = torch.tensor(value, dtype=torch.float32).log_softmax(-1)
    if int(os.environ.get('YOMO_MOBILE_ONLINE_PARTICLES', '0')):
        from online_particles import install
        install(main, geometry, assets, int(os.environ['YOMO_MOBILE_ONLINE_PARTICLES']))
    return (main, geometry)
