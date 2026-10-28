import torch
from torch import nn

class InitialEvidencePosterior(nn.Module):

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.ap = nn.Sequential(nn.Linear(21, 32), nn.SiLU(), nn.Linear(32, 24), nn.SiLU())
        self.readout = nn.Sequential(nn.Linear(60, 32), nn.SiLU(), nn.Linear(32, 12))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, rssi, mask, field, noise_variance, mean, covariance, ap_embedding):
        basis = field['basis']
        valid = mask.float().expand_as(field['mean'])
        residual = rssi - field['mean']
        posterior_residual = residual - (basis @ mean[..., None]).squeeze(-1)
        scalars = torch.cat((torch.stack((residual / 10, posterior_residual / 10, noise_variance.clamp_min(0.25).log() / 5), -1), basis, field['gradient'] / 10, (field['mean'] / 50)[..., None], valid[..., None]), -1)
        h = self.ap(torch.cat((scalars, ap_embedding.expand(*scalars.shape[:-1], 8)), -1))
        weight = valid[..., None]
        den = weight.sum(-2).clamp_min(1)
        average = (h * weight).sum(-2) / den
        spread = (((h - average[..., None, :]).square() * weight).sum(-2) / den + 1e-06).sqrt()
        raw = self.readout(torch.cat((average, spread, mean / 10, covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0).log1p() / 5), -1)).float()
        shift = 2 * raw[..., :6].tanh()
        scale = (0.6931471805599453 * raw[..., 6:].tanh()).exp() if self.mode == 'full' else torch.ones_like(shift)
        root = torch.linalg.cholesky(covariance)
        mapped_mean = mean + (root @ shift[..., None]).squeeze(-1)
        mapped_root = root * scale[..., None, :]
        mapped_covariance = mapped_root @ mapped_root.transpose(-1, -2)
        self.last_shift = shift.detach()
        self.last_scale = scale.detach()
        return (mapped_mean, mapped_covariance)

    def window(self, rssi, mask, field, noise_variance, mean, covariance, ap_embedding, policy):
        basis = field['basis']
        valid = mask.float().expand_as(field['mean'])
        residual = rssi - field['mean']
        posterior_residual = residual - torch.einsum('...tad,...d->...ta', basis, mean)
        scalars = torch.cat((torch.stack((residual / 10, posterior_residual / 10, noise_variance.clamp_min(0.25).log() / 5), -1), basis, field['gradient'] / 10, (field['mean'] / 50)[..., None], valid[..., None]), -1)
        inputs = torch.cat((scalars, ap_embedding.expand(*scalars.shape[:-1], 8)), -1)
        if policy == 'time_average':
            den = valid.sum(-2).clamp_min(1)
            inputs = (inputs * valid[..., None]).sum(-3) / den[..., None]
            valid = (valid.sum(-2) > 0).float()
        elif policy != 'evidence_set':
            raise ValueError('Unknown initial evidence pool')
        h = self.ap(inputs)
        if policy == 'evidence_set':
            h = h.flatten(-3, -2)
            valid = valid.flatten(-2, -1)
        weight = valid[..., None]
        den = weight.sum(-2).clamp_min(1)
        average = (h * weight).sum(-2) / den
        spread = (((h - average[..., None, :]).square() * weight).sum(-2) / den + 1e-06).sqrt()
        raw = self.readout(torch.cat((average, spread, mean / 10, covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0).log1p() / 5), -1)).float()
        shift = 2 * raw[..., :6].tanh()
        scale = (0.6931471805599453 * raw[..., 6:].tanh()).exp() if self.mode == 'full' else torch.ones_like(shift)
        root = torch.linalg.cholesky(covariance)
        mapped_mean = mean + (root @ shift[..., None]).squeeze(-1)
        mapped_root = root * scale[..., None, :]
        mapped_covariance = mapped_root @ mapped_root.transpose(-1, -2)
        self.last_shift = shift.detach()
        self.last_scale = scale.detach()
        return (mapped_mean, mapped_covariance)

def window_posterior(calibration, rssi, mask, field, noise, policy, ap_embedding):
    information, vector, _, _, _ = calibration.block_information(field['basis'], rssi - field['mean'], noise, mask)
    count = mask.any(-1).sum(-1).clamp_min(1).float()
    information = information / count[..., None, None]
    vector = vector / count[..., None]
    mean, covariance = calibration.posterior(information, vector)
    posterior = calibration.initial_encoder.window(rssi, mask, field, noise.diagonal(dim1=-2, dim2=-1), mean, covariance, ap_embedding, policy)
    return (information, vector, posterior)
