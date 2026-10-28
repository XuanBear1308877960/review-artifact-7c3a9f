import torch

def basin_probability(points, radio_score, prior_mean, prior_covariance, posterior, policy, cell_size=0.6, max_peaks=4, log_prominence=3.0):
    if policy not in ('density_mode', 'mass_mode'):
        raise ValueError('Unknown spatial basin projection')
    n = points.shape[-2]
    prefix = posterior.shape[:-1]
    with torch.no_grad():
        location = points.detach()
        difference = location - prior_mean.detach().unsqueeze(-2)
        precision = torch.linalg.inv(prior_covariance.detach())
        density = radio_score.detach() - 0.5 * torch.einsum('...nd,...de,...ne->...n', difference, precision, difference)
        density = density.masked_fill(posterior.detach() <= 0, -torch.inf)
        distances = torch.cdist(location, location, p=2, compute_mode='donot_use_mm_for_euclid_dist').square()
        neighbours = distances.topk(min(9, n), largest=False, dim=-1).indices
        nearby = torch.gather(density.unsqueeze(-2).expand(*prefix, n, n), -1, neighbours)
        local_maximum = density >= nearby.max(-1).values
        remaining = density.masked_fill(~local_maximum, -torch.inf)
        highest = density.max(-1).values
        columns = []
        available = []
        for _ in range(max_peaks):
            value, index = remaining.max(-1)
            valid = torch.isfinite(value) & (value >= highest - log_prominence)
            column = distances.gather(-1, index[..., None, None].expand(*prefix, n, 1)).squeeze(-1)
            columns.append(column)
            available.append(valid)
            remaining = remaining.masked_fill(column < (1.5 * cell_size) ** 2, -torch.inf)
        valid = torch.stack(available, -1)
        distance_to_peaks = torch.stack(columns, -1).masked_fill(~valid.unsqueeze(-2), torch.inf)
        nearest = distance_to_peaks.argmin(-1)
        mass = torch.zeros(*prefix, max_peaks, device=posterior.device, dtype=posterior.dtype)
        mass.scatter_add_(-1, nearest, posterior.detach())
        chosen = torch.zeros_like(highest, dtype=torch.long) if policy == 'density_mode' else mass.argmax(-1)
        selected = nearest == chosen.unsqueeze(-1)
        selected_mass = (posterior.detach() * selected).sum(-1)
        informative = radio_score.detach().max(-1).values - radio_score.detach().min(-1).values > 1e-05
        use = informative & (selected_mass > 1e-08)
        count = torch.where(informative, valid.sum(-1), torch.ones_like(chosen))
    restricted = posterior * selected
    restricted = restricted / restricted.sum(-1, keepdim=True).clamp_min(1e-30)
    probability = torch.where(use[..., None], restricted, posterior)
    return (probability, dict(peak_count=count, selected_basin_probability=torch.where(use, selected_mass, torch.ones_like(selected_mass))))
