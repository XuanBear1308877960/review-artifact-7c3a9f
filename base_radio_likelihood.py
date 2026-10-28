import math
import torch
from base_model import stabilize

def robust_radio_score(nominal_score, rssi, mask, prediction, covariance, outlier_probability, head_potential=None):
    valid = mask.float()
    probability = (1 - (1 - outlier_probability * valid).prod(-1)).clamp(1e-06, 1 - 1e-06)
    wide = stabilize(covariance + torch.diag_embed(100.0 * valid))
    chol = torch.linalg.cholesky(wide)
    residual = (rssi - prediction) * valid
    white = torch.linalg.solve_triangular(chol, residual[..., None], upper=False).squeeze(-1)
    quadratic = white.square().sum(-1)
    logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    dimensions = valid.sum(-1)
    nu = 3.0
    tail = torch.lgamma((nu + dimensions) / 2) - math.lgamma(nu / 2) - 0.5 * (dimensions * math.log(nu * math.pi) + logdet) - (nu + dimensions) / 2 * torch.log1p(quadratic / nu)
    normal_component = torch.log1p(-probability) + nominal_score
    tail_component = probability.log() + tail
    mixture = torch.logaddexp(normal_component, tail_component)
    responsibility = (normal_component - mixture).exp()
    if head_potential is not None:
        baseline = torch.logaddexp(normal_component - head_potential, tail_component)
        return (mixture, responsibility, baseline)
    return (mixture, responsibility)
