import math
import torch
from base_model import gaussian_log_prob, stabilize

def predictive_transfer(rssi, mask, field, device_mean, device_covariance, offset, sigma, outlier, pose_covariance, update=False, robust=False):
    basis = field['basis']
    valid = mask.float()
    noise = torch.diag_embed(field['map_variance'] + sigma.square()) + field['gradient'] @ pose_covariance @ field['gradient'].transpose(-1, -2)
    if not robust:
        scale = (1 - outlier).clamp_min(0.05).rsqrt()
        noise = noise * scale[..., None] * scale[..., None, :]
    design = basis * valid[..., None]
    noise = noise * valid[..., None] * valid[..., None, :] + torch.diag_embed(1 - valid)
    prediction = field['mean'] + (basis @ device_mean[..., None]).squeeze(-1) + offset
    residual = (rssi - prediction) * valid
    covariance = stabilize(design @ device_covariance @ design.transpose(-1, -2) + noise)
    score = gaussian_log_prob(residual, covariance) + 0.5 * (1 - valid).sum(-1) * math.log(2 * math.pi)
    responsibility = None
    if robust:
        from base_radio_likelihood import robust_radio_score
        score, responsibility = robust_radio_score(score, rssi, mask, prediction, covariance, outlier)
    if not update:
        return score
    if responsibility is not None:
        noise = noise / responsibility.clamp_min(0.001)[..., None, None]
        covariance = stabilize(design @ device_covariance @ design.transpose(-1, -2) + noise)
    cross = device_covariance @ design.transpose(-1, -2)
    gain = torch.linalg.solve(covariance, cross.transpose(-1, -2)).transpose(-1, -2)
    mean = device_mean + (gain @ residual[..., None]).squeeze(-1)
    transform = torch.eye(6, device=basis.device) - gain @ design
    covariance = stabilize(transform @ device_covariance @ transform.transpose(-1, -2) + gain @ noise @ gain.transpose(-1, -2))
    return (score, mean, covariance)
