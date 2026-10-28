import math
import numpy as np
import torch
from scipy.special import logsumexp

class ForwardCS:

    def __init__(self, prior):
        from trajectory_raw_hmm import unpack
        self.prior = prior
        self.p = unpack(prior['raw_HMM_parameters'])
        self.alpha = {}
        self.times = {}

    def update(self, engine, event):
        from base_quadrature import position_support
        from base_model import stabilize
        from trajectory_raw_hmm import block_context
        from trajectory_joint_RF import auxiliary_emission
        p = event['payload']
        anchor = p['anchor_id']
        if anchor not in engine.cs_history:
            return
        a = engine.anchor_names.index(anchor)
        now = engine.timestamp / 1000000000.0
        raw = float(p['raw_range_m'])
        seq = engine.cs_history[anchor]
        seq.append((now, raw, float(p['range_m']), event['observation_id']))
        recent = [v[1] for v in seq if now - v[0] <= 1.5]
        if len(recent) < 3:
            recent = [v[1] for v in seq if now - v[0] <= 3.0]
        mad = 1.4826 * np.median(abs(np.array(recent) - np.median(recent)))
        context = block_context(a, [now], [raw], [0.35 + min(float(mad), 10.0)], self.prior['raw_HMM_parameters'], self.prior['raw_HMM_background_range_m'])
        if 'CS_auxiliary_values' in p:
            values = np.array(p['CS_auxiliary_values'], float)
            finite = np.array(p['CS_auxiliary_is_finite'], bool)
            valid = finite[:2] & (values[:2] > 0)
            valid[1] &= values[3] > 0
            context.update(auxiliary_parameters=self.prior['joint_RF_auxiliary_parameters'], auxiliary_values=values[None, :2], auxiliary_valid=valid[None])
        import os
        rule = os.environ.get('YOMO_MOBILE_ONLINE_CS_QUADRATURE', 'hermite')
        points, logprior, g = position_support(engine.mean[:, :2], stabilize(engine.cov[:, :2, :2]), engine.grid_xy, rule)
        if rule == 'hermite':
            points = points[:, g:]
            logprior = logprior[:, g:]
        distance = (points - engine.anchors[a]).square().sum(-1).sqrt().numpy()
        K, N = distance.shape
        sigma = context['sigma'][0]
        parameter = self.p
        error = raw - parameter['gain'] * distance[..., None] - parameter['anchor_mean'][a] - parameter['alias']
        error = np.concatenate((error, np.full((K, N, 1), raw - context['background'])), -1)
        emission = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi) - np.log(sigma) - 2 * np.log1p(error ** 2 / (3 * sigma ** 2))
        if 'auxiliary_parameters' in context:
            ctx = dict(context, auxiliary_values=np.repeat(context['auxiliary_values'], K * N, axis=0), auxiliary_valid=np.repeat(context['auxiliary_valid'], K * N, axis=0))
            emission += auxiliary_emission(distance.ravel(), error.reshape(-1, 4), ctx).reshape(K, N, 4)
        old = self.alpha.get(anchor, np.broadcast_to(parameter['probability'], (K, 4)).copy())
        if anchor in self.times:
            dt = max(1e-05, now - self.times[anchor])
            stick = np.exp(-dt / parameter['persistence_s'])
            transition = (1 - stick[:, None]) * parameter['probability'][None] + np.diag(stick)
            old = old @ transition
        joint = logprior.numpy()[..., None] + np.log(old.clip(1e-30))[:, None] + emission
        evidence = logsumexp(joint, axis=(1, 2))
        posterior = np.exp(logsumexp(joint, axis=-1) - evidence[:, None])
        self.alpha[anchor] = np.exp(logsumexp(joint, axis=1) - evidence[:, None])
        self.times[anchor] = now
        w = torch.tensor(posterior, dtype=torch.float32)
        xy = (w[..., None] * points).sum(1)
        d = points - xy[:, None]
        P = torch.einsum('kn,knd,kne->kde', w, d, d)
        oldP = stabilize(engine.cov[:, :2, :2])
        gain = torch.linalg.solve(oldP, engine.cov[:, :, :2].transpose(-1, -2)).transpose(-1, -2)
        engine.mean += torch.einsum('kdz,kz->kd', gain, xy - engine.mean[:, :2])
        engine.cov = stabilize(engine.cov + gain @ (P - oldP) @ gain.transpose(-1, -2))
        engine.log_weight = (engine.log_weight + torch.tensor(evidence, dtype=torch.float32)).log_softmax(-1)
        engine.cs_diagnostics.append(dict(observation_id=event['observation_id'], causal_raw_HMM=True, return_probability=self.alpha[anchor].tolist(), auxiliary_channels_used='auxiliary_parameters' in context))
