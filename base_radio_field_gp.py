import math
import numpy as np
import torch
from torch import nn

def inducing_indices(xy, count):
    count = min(int(count), len(xy))
    selected = [int(((xy - xy.mean(0)) ** 2).sum(-1).argmin())]
    distance = np.full(len(xy), np.inf)
    while len(selected) < count:
        distance = np.minimum(distance, ((xy - xy[selected[-1]]) ** 2).sum(-1))
        distance[selected] = -1
        selected.append(int(distance.argmax()))
    return np.asarray(selected)

def matern32(left, right, lengthscale):
    radius = np.linalg.norm(left[:, None] - right[None], axis=-1)
    scaled = math.sqrt(3) * radius / lengthscale
    return (1 + scaled) * np.exp(-scaled)

class SparsePhysicalSource(nn.Module):

    def __init__(self, cache, geometry, inducing_count=32, lengthscale=1.2):
        super().__init__()
        self.lengthscale = float(lengthscale)
        xy = np.asarray(cache['xy'], float)
        mask = np.asarray(cache['mask'], bool)
        mean = np.asarray(cache['trend'], float) + np.asarray(cache['residual'], float)
        variance = np.asarray(cache['var'], float)
        centres = np.asarray(geometry['centres_m'], float)
        exponent = np.asarray(geometry['exponents'], float)
        exponent = np.clip(exponent, 0.2, 6.0)
        log_distance = 0.5 * np.log10(((xy[:, None] - centres) ** 2).sum(-1) + 1.0)
        power = np.asarray([np.mean(np.asarray(cache['trend'])[mask[:, a], a] + 10 * exponent[a] * log_distance[mask[:, a], a]) for a in range(mean.shape[-1])])
        trend = power - 10 * exponent * log_distance
        indices = inducing_indices(xy, inducing_count)
        inducing = xy[indices]
        m = len(inducing)
        base_uu = matern32(inducing, inducing, self.lengthscale)
        base_xu = matern32(xy, inducing, self.lengthscale)
        roots = []
        coefficients = []
        adjustments = []
        amplitudes = []
        noise = []
        fit_error = []
        for a in range(mean.shape[-1]):
            valid = mask[:, a] & np.isfinite(mean[:, a]) & np.isfinite(variance[:, a])
            if valid.sum() < 4:
                raise ValueError('OPPO map has insufficient observations for an AP')
            residual = mean[valid, a] - trend[valid, a]
            amplitude = max(2.0, float(np.std(residual)))
            amp2 = amplitude ** 2
            kuu = amp2 * base_uu + np.eye(m) * (amp2 * 0.0001 + 1e-06)
            root = np.linalg.cholesky(kuu)
            inverse_root_transpose = np.linalg.solve(root, np.eye(m)).T
            feature = amp2 * base_xu[valid] @ inverse_root_transpose
            measurement = np.maximum(variance[valid, a], 0.25)
            diagonal = measurement + np.maximum(amp2 - (feature ** 2).sum(-1), 0.0)
            precision = np.eye(m) + feature.T @ (feature / diagonal[:, None])
            posterior = np.linalg.inv(precision)
            posterior = 0.5 * (posterior + posterior.T)
            coefficient = posterior @ (feature.T @ (residual / diagonal))
            roots.append(inverse_root_transpose)
            coefficients.append(coefficient)
            adjustments.append(posterior - np.eye(m))
            amplitudes.append(amp2)
            noise.append(float(np.median(measurement)))
            fit_error.append(float(np.mean(abs(feature @ coefficient - residual))))
        values = dict(inducing=inducing, centres=centres, exponent=exponent, power=power, inverse_root_transpose=np.asarray(roots), coefficient=np.asarray(coefficients), posterior_adjustment=np.asarray(adjustments), amplitude_squared=np.asarray(amplitudes), measurement_noise=np.asarray(noise), inducing_visibility=np.asarray(cache['visibility'])[indices].T)
        for name, value in values.items():
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
        self.fit_summary = dict(source='OPPO reference map only', inducing_count=m, lengthscale_m=self.lengthscale, kernel='Matern_3_2', residual='Sparse FITC Gaussian process', AP_fit_MAE_db=fit_error, source_measurement_variance_db2=noise, target_inputs=False)

    def forward(self, points, component_gradient=False):
        offset = points[..., None, :] - self.inducing
        radius = offset.square().sum(-1).clamp_min(1e-12).sqrt()
        scaled = math.sqrt(3) * radius / self.lengthscale
        decay = (-scaled).exp()
        base = (1 + scaled) * decay
        kernel = base[..., None, :] * self.amplitude_squared[:, None]
        feature = torch.einsum('...am,amn->...an', kernel, self.inverse_root_transpose)
        derivative = -3 / self.lengthscale ** 2 * offset * decay[..., None]
        kernel_derivative = derivative[..., None, :, :] * self.amplitude_squared[:, None, None]
        feature_derivative = torch.einsum('...amz,amn->...anz', kernel_derivative, self.inverse_root_transpose)
        residual = torch.einsum('...am,am->...a', feature, self.coefficient)
        residual_gradient = torch.einsum('...amz,am->...az', feature_derivative, self.coefficient)
        radio_offset = points[..., None, :] - self.centres
        radius2 = radio_offset.square().sum(-1) + 1.0
        trend = self.power - 5 * self.exponent * radius2.log10()
        trend_gradient = -10 / math.log(10) * self.exponent[..., None] * radio_offset / radius2[..., None]
        raw_variance = self.amplitude_squared + self.measurement_noise + torch.einsum('...am,amn,...an->...a', feature, self.posterior_adjustment, feature)
        variance = raw_variance.clamp_min(0.25)
        variance_gradient = 2 * torch.einsum('...amz,amn,...an->...az', feature_derivative, self.posterior_adjustment, feature)
        variance_gradient = variance_gradient * (raw_variance > 0.25)[..., None]
        total = base.sum(-1).clamp_min(1e-08)
        visibility = torch.einsum('...m,am->...a', base, self.inducing_visibility) / total[..., None]
        visibility_gradient = torch.einsum('...mz,am->...az', derivative, self.inducing_visibility) / total[..., None, None] - visibility[..., None] * derivative.sum(-2)[..., None, :] / total[..., None, None]
        visibility_gradient = visibility_gradient * ((visibility > 0.01) & (visibility < 0.99))[..., None]
        result = dict(trend=trend, residual=residual, variance=variance, visibility=visibility.clamp(0.01, 0.99), mean=trend + residual, gradient=trend_gradient + residual_gradient)
        if component_gradient:
            result['component_gradient'] = torch.stack((trend_gradient, residual_gradient, variance_gradient, visibility_gradient), -2)
        return result
