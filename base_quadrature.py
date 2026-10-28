import math
import numpy as np
import torch
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(5)
_GH_Z = np.stack(np.meshgrid(_GH_X, _GH_X, indexing='ij'), -1).reshape(-1, 2) * math.sqrt(2.0)
_GH_LOG_W = np.log(np.outer(_GH_W, _GH_W).reshape(-1) / math.pi)

def position_support(mean, covariance, source_xy, rule='mixed', log_area=None):
    shape = mean.shape[:-1]
    g = len(source_xy)
    chol = torch.linalg.cholesky(covariance)
    global_points = source_xy.expand(*shape, g, 2)
    if rule == 'hermite':
        z = torch.as_tensor(_GH_Z, device=mean.device, dtype=mean.dtype)
        local = mean[..., None, :] + torch.einsum('...de,ne->...nd', chol, z)
        points = torch.cat((global_points, local), -2)
        local_prior = torch.as_tensor(_GH_LOG_W, device=mean.device, dtype=mean.dtype).expand(*shape, 25)
        prior = torch.cat((torch.full((*shape, g), -torch.inf, device=mean.device, dtype=mean.dtype), local_prior), -1)
    elif rule in ('mixed', 'balanced'):
        difference = global_points - mean[..., None, :]
        precision = torch.linalg.inv(covariance)
        density = -0.5 * torch.einsum('...nd,...de,...ne->...n', difference, precision, difference)
        if log_area is not None:
            density = density + log_area
        grid_prior = density.log_softmax(-1)
        fraction = ((covariance.diagonal(dim1=-2, dim2=-1).mean(-1).sqrt() - 0.15) / 0.45).clamp(0.0, 0.75)
        local_mean = mean
        local_chol = chol
        if rule == 'balanced':
            probability = grid_prior.exp()
            grid_mean = (probability[..., None] * global_points).sum(-2)
            centred = global_points - grid_mean[..., None, :]
            grid_cov = torch.einsum('...n,...nd,...ne->...de', probability, centred, centred)
            shift = grid_mean - mean
            white_shift = torch.linalg.solve_triangular(chol, shift[..., None], upper=False).squeeze(-1)
            left = torch.linalg.solve_triangular(chol, grid_cov, upper=False)
            white_cov = torch.linalg.solve_triangular(chol, left.transpose(-1, -2), upper=False).transpose(-1, -2)
            white_cov = 0.5 * (white_cov + white_cov.transpose(-1, -2))
            largest = 0.5 * (white_cov[..., 0, 0] + white_cov[..., 1, 1]) + (0.25 * (white_cov[..., 0, 0] - white_cov[..., 1, 1]).square() + white_cov[..., 0, 1].square()).clamp_min(1e-16).sqrt()
            bound = 0.9
            value = largest + white_shift.square().sum(-1) + bound
            root = 2 * bound / (value + (value.square() - 4 * largest * bound).clamp_min(0).sqrt()).clamp_min(1e-12)
            fraction = torch.minimum(fraction, 0.95 * root)
            remainder = (1 - fraction).clamp_min(1e-06)
            local_mean = mean - fraction[..., None] / remainder[..., None] * shift
            outer = shift[..., None] * shift[..., None, :]
            residual = (covariance - fraction[..., None, None] * grid_cov - (fraction / remainder)[..., None, None] * outer) / remainder[..., None, None]
            residual = 0.5 * (residual + residual.transpose(-1, -2))
            local_chol = torch.linalg.cholesky(residual)
        local = torch.stack((torch.zeros_like(mean), 3 ** 0.5 * local_chol[..., 0], -3 ** 0.5 * local_chol[..., 0], 3 ** 0.5 * local_chol[..., 1], -3 ** 0.5 * local_chol[..., 1]), -2) + local_mean[..., None, :]
        points = torch.cat((global_points, local), -2)
        local_prior = torch.tensor([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6], device=mean.device, dtype=mean.dtype).log().expand(*shape, 5)
        prior = torch.cat((grid_prior + fraction.clamp_min(1e-20).log()[..., None], local_prior + torch.log1p(-fraction)[..., None]), -1)
    else:
        raise ValueError('Unknown position rule: ' + rule)
    return (points, prior, g)
