import math
import numpy as np
import torch
from scipy.special import softmax, logsumexp

class VelocityStates:

    def __init__(self, assets):
        from speed_magnitude import SpeedContext
        self.context = SpeedContext()
        self.head = assets['speed_head']
        self.prior = assets['prior']
        self.p = np.asarray(self.prior['velocity_state_parameters'])
        self.alpha = None
        self.last_ns = None
        z, w = np.polynomial.hermite.hermgauss(5)
        self.z = np.stack(np.meshgrid(z, z, indexing='ij'), -1).reshape(-1, 2) * math.sqrt(2)
        self.logw = np.log(np.outer(w, w).ravel() / math.pi)

    def update(self, engine, event):
        from base_model import stabilize
        self.context.observe(event)
        now = event['monotonic_timestamp_ns']
        if self.last_ns is None:
            self.last_ns = now
            return
        dt = (now - self.last_ns) / 1000000000.0
        if dt < 0.5:
            return
        feature = self.context.features(now)
        if feature[6] < 0.5:
            return
        speed, sigma = self.head(torch.tensor(feature))
        speed = float(speed)
        self.last_ns = now
        p = self.p
        groups = softmax([0.0, p[4], p[5]])
        prob = np.array([groups[0] / 2, groups[0] / 2, groups[1] / 2, groups[1] / 2, groups[2]])
        old = np.broadcast_to(prob, (8, 5)).copy() if self.alpha is None else self.alpha
        stick = np.exp(-dt / np.exp(p[[6, 6, 7, 7, 8]]))
        T = (1 - stick[:, None]) * prob[None] + np.diag(stick)
        old = old @ T
        P = stabilize(engine.cov[:, 2:4, 2:4])
        root = torch.linalg.cholesky(P).numpy()
        v = engine.mean[:, None, 2:4].numpy() + np.einsum('kdz,nz->knd', root, self.z)
        angle = np.arange(8) * math.pi / 4
        c = np.cos(angle)
        s = np.sin(angle)
        rotated = np.stack((v[..., 0] * c[:, None] + v[..., 1] * s[:, None], -v[..., 0] * s[:, None] + v[..., 1] * c[:, None]), -1)
        means = np.array([[speed, 0], [-speed, 0], [0, math.exp(p[3]) * speed], [0, -math.exp(p[3]) * speed], [0, 0]])
        sp, st, ss = np.exp(p[:3])
        scales = np.array([[sp, st], [sp, st], [st, sp], [st, sp], [ss, ss]]) / math.sqrt(dt)
        residual = rotated[:, :, None] - means
        constant = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi)
        emission = (constant - np.log(scales) - 2 * np.log1p(residual ** 2 / (3 * scales ** 2))).sum(-1)
        joint = emission + np.log(old.clip(1e-30))[:, None] + self.logw[None, :, None]
        evidence = logsumexp(joint, axis=(1, 2))
        mass = np.exp(logsumexp(joint, axis=-1) - evidence[:, None])
        self.alpha = np.exp(logsumexp(joint, axis=1) - evidence[:, None])
        w = torch.tensor(mass, dtype=torch.float32)
        points = torch.tensor(v, dtype=torch.float32)
        mean = (w[..., None] * points).sum(1)
        d = points - mean[:, None]
        newP = torch.einsum('kn,knd,kne->kde', w, d, d)
        gain = torch.linalg.solve(P, engine.cov[:, :, 2:4].transpose(-1, -2)).transpose(-1, -2)
        engine.mean += torch.einsum('kdz,kz->kd', gain, mean - engine.mean[:, 2:4])
        engine.cov = stabilize(engine.cov + gain @ (newP - P) @ gain.transpose(-1, -2))
        engine.log_weight = (engine.log_weight + torch.tensor(evidence, dtype=torch.float32)).log_softmax(-1)
