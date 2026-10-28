from collections import deque
import numpy as np

class RadioNoise:

    def __init__(self, parameters):
        self.p = np.asarray(parameters)
        self.rows = [deque(maxlen=16) for _ in range(4)]

    def covariance(self, a, left, right):
        t, w = left
        u, v = right
        gap = abs(t[:, None] - u[None, :])
        colored = np.exp(2 * self.p[4 + a]) * np.exp(-gap / np.exp(self.p[8]))
        shared = np.exp(2 * self.p[9 + a]) * (gap < 1e-08)
        return float(w @ (colored + shared) @ v)

    def prepare(self, event, y, mean, basis, pose_variance, mask, extra):
        meta = event['payload'].get('RSSI_contributors')
        descriptors = []
        if meta is None:
            for a in range(4):
                descriptors.append((np.array([event['monotonic_timestamp_ns'] / 1000000000.0]), np.ones(1)))
        else:
            for a in range(4):
                w = np.asarray(meta['weights'][a], float)
                t = np.asarray(meta['times_s'][a], float)
                valid = w > 0
                descriptors.append((t[valid], w[valid]))
        offset = np.zeros((8, 4))
        basis_offset = np.zeros((8, 4, 6))
        variance = np.zeros((8, 4))
        for a in range(4):
            descriptor = descriptors[a]
            past = list(self.rows[a])
            white = np.exp(2 * self.p[a]) + 0.1 ** 2
            current = self.covariance(a, descriptor, descriptor) + white + extra[a]
            variance[:, a] = current
            if past and len(descriptor[0]):
                n = len(past)
                R = np.array([[self.covariance(a, r['descriptor'], s['descriptor']) for s in past] for r in past])
                R += np.diag([white + r['extra'] for r in past])
                cross = np.array([self.covariance(a, descriptor, r['descriptor']) for r in past])
                joint = R[None] + np.array([r['pose'] for r in past]).T[:, :, None] * np.eye(n)[None]
                coefficient = np.linalg.solve(joint + np.eye(n) * 1e-07, np.broadcast_to(cross, (8, n))[..., None])[..., 0]
                residual = np.array([r['y'] - r['mean'] for r in past]).T
                offset[:, a] = (coefficient * residual).sum(-1)
                basis_offset[:, a] = np.einsum('kt,tkd->kd', coefficient, np.stack([r['basis'] for r in past]))
                variance[:, a] = (current - (coefficient * cross).sum(-1)).clip(0.05)
        for a in np.where(mask)[0]:
            self.rows[a].append(dict(descriptor=descriptors[a], y=float(y[a]), mean=mean[:, a].copy(), basis=basis[:, a].copy(), pose=pose_variance[:, a].copy(), extra=float(extra[a])))
        return (offset, basis_offset, variance)
