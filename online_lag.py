from collections import deque
import numpy as np

class SupportLag:

    def __init__(self, limit=12):
        self.rows = deque(maxlen=limit)

    def predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        for r in self.rows:
            r['cross'] = r['cross'] @ F.T

    def append(self, row, mean, covariance, motion, offset, basis_offset, projected):
        self.rows.append(dict(factor=row, xy=mean[:, :2].copy(), cov=covariance[:, :2, :2].copy(), cross=covariance[:, :2, :4].copy(), motion=motion.copy(), offset=offset.copy(), basis_offset=basis_offset.copy(), noise_without_pose=row['variance'].numpy() - projected))

    def observe(self, mean, covariance, new_mean, new_covariance):
        P = (covariance + covariance.transpose(0, 2, 1)) * 0.5 + np.eye(4) * 1e-06
        for r in self.rows:
            gain = np.linalg.solve(P, r['cross'].transpose(0, 2, 1)).transpose(0, 2, 1)
            r['xy'] += np.einsum('kdz,kz->kd', gain, new_mean - mean)
            r['cov'] += gain @ (new_covariance - covariance) @ gain.transpose(0, 2, 1)
            value, vector = np.linalg.eigh((r['cov'] + r['cov'].transpose(0, 2, 1)) * 0.5)
            r['cov'] = vector * np.maximum(value, 1e-06)[:, None] @ vector.transpose(0, 2, 1)
            r['cross'] = gain @ new_covariance
