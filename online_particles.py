import math
import numpy as np
import torch
from scipy.special import logsumexp

def install(main, geometry, assets, count):

    def wrap(engine, radio):
        base = type(engine)

        class ParticleEngine(base):

            def start_particles(self):
                from base_model import stabilize
                self.rng = np.random.default_rng(1737)
                self.particle_count = count
                P = stabilize(self.cov[:, :4, :4]).numpy()
                z = self.rng.normal(size=(8, count, 4))
                z -= z.mean(1, keepdims=True)
                self.particles = self.mean[:, None, :4].numpy() + np.einsum('kde,kne->knd', np.linalg.cholesky(P), z)
                self.particle_logw = np.full((8, count), -math.log(count))
                self.summarize()
                for key, value in self.forward_cs.alpha.items():
                    self.forward_cs.alpha[key] = np.repeat(value[:, None], count, axis=1)

            def summarize(self):
                from base_model import stabilize
                w = np.exp(self.particle_logw)
                mean = (w[..., None] * self.particles).sum(1)
                d = self.particles - mean[:, None]
                P = np.einsum('kn,knd,kne->kde', w, d, d) + np.eye(4) * 1e-05
                self.mean[:, :4] = torch.tensor(mean, dtype=torch.float32)
                self.cov[:, :4, 4:] = 0
                self.cov[:, 4:, :4] = 0
                self.cov[:, :4, :4] = torch.tensor(P, dtype=torch.float32)
                self.cov = stabilize(self.cov)

            def absorb(self, score):
                mass = self.particle_logw + score
                normalizer = logsumexp(mass, axis=-1)
                self.particle_logw = mass - normalizer[:, None]
                self.log_weight = (self.log_weight + torch.tensor(normalizer, dtype=torch.float32)).log_softmax(-1)
                self.summarize()
                for k in range(8):
                    w = np.exp(self.particle_logw[k])
                    if 1 / np.square(w).sum() < count * 0.5:
                        u = (self.rng.random() + np.arange(count)) / count
                        ix = np.searchsorted(np.cumsum(w), u).clip(0, count - 1)
                        self.particles[k] = self.particles[k, ix]
                        self.particle_logw[k] = -math.log(count)
                        for key in self.forward_cs.alpha:
                            self.forward_cs.alpha[key][k] = self.forward_cs.alpha[key][k, ix]

            def consume(self, event):
                result = super().consume(event)
                ready = self.steps >= 5 if hasattr(self, 'steps') else len(self.outputs) >= 5
                if self.particles is None and ready:
                    self.start_particles()
                return result

            def _advance(self, ns):
                dt = (int(ns) - self.timestamp) / 1000000000.0
                predicted = None
                if self.particles is not None and dt > 0:
                    state = np.repeat(self.mean.numpy()[:, None], count, axis=1).copy()
                    state[..., :4] = self.particles
                    flat = state.reshape(-1, self.dim)
                    x, Q = self.motion.predict_batch(flat, np.zeros((len(flat), self.dim, self.dim)), dt, self.filter_cfg)
                    C = Q[:, :4, :4] + np.eye(4) * 1e-12
                    z = self.rng.normal(size=(len(flat), 4))
                    predicted = (x[:, :4] + np.einsum('kde,ke->kd', np.linalg.cholesky(C), z)).reshape(8, count, 4)
                super()._advance(ns)
                if predicted is not None:
                    self.particles = predicted
                    self.summarize()
                    if hasattr(self, 'lag_prior'):
                        self.lag_prior = (self.mean[:, :4].numpy().copy(), self.cov[:, :4, :4].numpy().copy())

            def _consume_cs(self, event):
                if self.particles is None:
                    return super()._consume_cs(event)
                from trajectory_raw_hmm import block_context
                from trajectory_joint_RF import auxiliary_emission
                p = event['payload']
                anchor = p['anchor_id']
                a = self.anchor_names.index(anchor)
                now = self.timestamp / 1000000000.0
                raw = float(p['raw_range_m'])
                seq = self.cs_history[anchor]
                seq.append((now, raw, float(p['range_m']), event['observation_id']))
                recent = [v[1] for v in seq if now - v[0] <= 1.5]
                if len(recent) < 3:
                    recent = [v[1] for v in seq if now - v[0] <= 3.0]
                scale = 0.35 + min(10.0, 1.4826 * float(np.median(abs(np.array(recent) - np.median(recent)))))
                prior = assets['prior']
                parameter = self.forward_cs.p
                context = block_context(a, [now], [raw], [scale], prior['raw_HMM_parameters'], prior['raw_HMM_background_range_m'])
                distance = np.linalg.norm(self.particles[..., :2] - self.anchors[a].numpy(), axis=-1)
                error = raw - parameter['gain'] * distance[..., None] - parameter['anchor_mean'][a] - parameter['alias']
                error = np.concatenate((error, np.full((8, count, 1), raw - context['background'])), -1)
                sigma = context['sigma'][0]
                emission = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi) - np.log(sigma) - 2 * np.log1p(error ** 2 / (3 * sigma ** 2))
                if 'CS_auxiliary_values' in p:
                    v = np.array(p['CS_auxiliary_values'])
                    finite = np.array(p['CS_auxiliary_is_finite'], bool)
                    valid = finite[:2] & (v[:2] > 0)
                    valid[1] &= v[3] > 0
                    ctx = dict(context, auxiliary_parameters=prior['joint_RF_auxiliary_parameters'], auxiliary_values=np.broadcast_to(v[:2], (8 * count, 2)), auxiliary_valid=np.broadcast_to(valid, (8 * count, 2)))
                    emission += auxiliary_emission(distance.ravel(), error.reshape(-1, 4), ctx).reshape(8, count, 4)
                old = self.forward_cs.alpha.get(anchor, np.broadcast_to(parameter['probability'], (8, count, 4)).copy())
                if anchor in self.forward_cs.times:
                    dt = max(1e-05, now - self.forward_cs.times[anchor])
                    stick = np.exp(-dt / parameter['persistence_s'])
                    transition = (1 - stick[:, None]) * parameter['probability'][None] + np.diag(stick)
                    old = old @ transition
                joint = np.log(old.clip(1e-30)) + emission
                score = logsumexp(joint, axis=-1)
                self.forward_cs.alpha[anchor] = np.exp(joint - score[..., None])
                self.forward_cs.times[anchor] = now
                self.absorb(score)
                self.cs_diagnostics.append(dict(observation_id=event['observation_id'], causal_raw_HMM=True, particles=8 * count))

            def update_position(self, event):
                if self.particles is None:
                    return super().update_position(event)
                from base_spatial_geometry import calibrated_field
                from base_model import stabilize, gaussian_log_prob
                from base_data import radio_tokens
                y = torch.zeros(4)
                mask = torch.zeros(4, dtype=torch.bool)
                for ap in event['payload']['ap_observations']:
                    a = self.lookup[ap['ap_id_hash']]
                    y[a] = ap['rssi_dbm']
                    mask[a] = True
                encoded = self._encoded(mask)
                mu, C = self.active_posterior
                g = len(self.grid_xy)
                points = torch.cat((self.grid_xy[None].expand(8, -1, 2), torch.tensor(self.particles[..., :2], dtype=torch.float32)), 1)
                n = points.shape[1]
                field = self.net.field(points, encoded['motion'][0].expand(8, n, 32))
                field = calibrated_field(field, mu[:, None])
                B = field['basis']
                prediction = field['mean'] + (B @ mu[:, None, :, None])[..., 0]
                mapping = B @ C[:, None] @ B.transpose(-1, -2)
                covariance = mapping + torch.diag_embed(field['map_variance'] + encoded['sigma'][0].square())
                valid = mask.float()
                covariance = covariance * valid[:, None] * valid[None] + torch.diag(1 - valid)
                old = gaussian_log_prob((y - prediction) * valid, stabilize(covariance)) + 0.5 * (~mask).sum() * math.log(2 * math.pi)
                potential = self.net.localization(y.expand(8, n, 4), mask.expand(8, n, 4), prediction, covariance.diagonal(dim1=-2, dim2=-1), field, mapping.diagonal(dim1=-2, dim2=-1), self.net.field.ap_embedding.weight, None, None, covariance, None)
                score = old + potential
                self.absorb(score[:, g:].numpy())
                self.last_wifi_distribution = dict(points=points[:, :g].clone(), probability=score[:, :g].softmax(-1), time=self.timestamp / 1000000000.0)
                self.rssi_history.append(y.numpy())
                self.mask_history.append(mask.numpy())
                self.times.append(self.timestamp / 1000000000.0)
                token = radio_tokens(np.asarray(self.rssi_history), np.asarray(self.mask_history), np.asarray(self.times))[-1]
                self.radio.append(token)
                weight = self.log_weight.softmax(-1)
                out = dict(observation_id=event['observation_id'], timestamp_ns=self.timestamp, xy_m=(weight[:, None] * self.mean[:, :2]).sum(0).tolist(), particles=8 * count)
                self.outputs.append(out)
                self.forecasts.append(dict(observation_id=event['observation_id'], timestamp_ns=self.timestamp))
                return out
        engine.__class__ = ParticleEngine
        engine.particles = None
    wrap(main, True)
    wrap(geometry, False)
