import torch
from torch import nn
from torch.nn import functional as F

def mlp_with_directions(network, inputs, directions):
    value = inputs
    jacobian = directions
    for layer in network:
        if isinstance(layer, nn.Linear):
            value = F.linear(value, layer.weight, layer.bias)
            jacobian = torch.einsum('...id,oi->...od', jacobian, layer.weight)
        elif isinstance(layer, nn.SiLU):
            probability = value.sigmoid()
            jacobian = jacobian * (probability * (1 + value * (1 - probability)))[..., None]
            value = F.silu(value)
        else:
            raise TypeError('Unsupported physical-basis activation: ' + type(layer).__name__)
    result = value.tanh().float()
    return (result, jacobian * (1 - result.square())[..., None])

def calibrated_field(field, device_mean):
    if 'basis_gradient' not in field:
        return field
    source = field.get('source_gradient', field['gradient'])
    gradient = source + torch.einsum('...adz,...d->...az', field['basis_gradient'], device_mean)
    return dict(field, source_gradient=source, gradient=gradient)

def source_tangent_prior(gradient, source, visibility, mask):
    square = source.square().sum(-1)
    gain = (gradient * source).sum(-1) / square.clamp_min(1e-06)
    orthogonal = gradient - gain[..., None] * source
    penalty = orthogonal.square().sum(-1) / (square + 0.25 ** 2) + F.relu(0.65 - gain).square() + 0.1 * F.relu(gain - 2).square()
    weight = visibility.detach() * square.detach() / (square.detach() + 0.25 ** 2) * mask.float()
    denominator = weight.sum().clamp_min(1)
    loss = (penalty * weight).sum() / denominator
    alignment = (gradient * source).sum(-1) / (gradient.square().sum(-1) * square).clamp_min(1e-08).sqrt()
    return (loss, (gain.detach() * weight).sum() / denominator, (alignment.detach() * weight).sum() / denominator)

class GeometryRSSIPotential(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, rssi, mask, prediction, field, covariance, cholesky, device_variance):
        valid = mask.float()
        residual = (rssi - prediction) * valid
        white = torch.linalg.solve_triangular(cholesky, residual[..., None], upper=False).squeeze(-1)
        gradient = field['gradient'] * valid[..., None]
        design = torch.linalg.solve_triangular(cholesky, gradient, upper=False)
        gram = design.transpose(-1, -2) @ design
        rhs = (design.transpose(-1, -2) @ white[..., None]).squeeze(-1)
        displacement = torch.linalg.solve(gram + 0.01 * torch.eye(2, device=rssi.device), rhs[..., None]).squeeze(-1)
        tangent = (design @ displacement[..., None]).squeeze(-1)
        orthogonal = white - tangent
        source = field.get('source_gradient', field['gradient']) * valid[..., None]
        source = torch.linalg.solve_triangular(cholesky, source, upper=False)
        alignment = (source * design).sum((-1, -2)) / (source.square().sum((-1, -2)) * design.square().sum((-1, -2))).clamp_min(1e-08).sqrt()
        attenuation = (design.square().sum((-1, -2)) + 0.01) / (source.square().sum((-1, -2)) + 0.01)
        scalars = torch.stack((orthogonal.square().sum(-1).log1p(), tangent.square().sum(-1).log1p(), gram[..., 0, 0].log1p(), gram[..., 1, 1].log1p(), gram[..., 0, 1] / (1 + gram.diagonal(dim1=-2, dim2=-1).sum(-1)), alignment, attenuation.log().clamp(-5, 5), device_variance.mean(-1).log1p() / 5), -1)
        features = torch.cat((white.clamp(-20, 20) / 5, valid, scalars), -1)
        if features.shape[-1] != 16:
            raise ValueError('Geometry head expects the four-map-AP protocol')
        return 4 * self.network(features).squeeze(-1).tanh()
