import torch

def features(gradient, rssi, mask, delta, context, count):
    lags = [0.9] if count == 1 else [0.35, 0.9, 1.8]
    b, k, q, a, _ = gradient.shape
    age = context['now_time'][:, None] - context['times']
    target = age.new_tensor(lags)
    time_weight = torch.exp(-0.5 * ((age[..., None] - target) / (0.1 + 0.2 * target)).square())
    time_weight = time_weight * ((age >= 0) & (age <= 3.0))[..., None]
    weight = time_weight[:, :, None] * context['mask'][..., None]
    mass = weight.sum(1)
    weight = weight / mass.clamp_min(1e-08)[:, None]
    past_rssi = torch.einsum('bhal,bha->bal', weight, context['rssi'])
    past_delta = torch.einsum('bhal,bhka->bkal', weight, context['delta'])
    past_path = torch.einsum('bhal,bhkz->bkalz', weight, context['path'])
    past_variance = torch.einsum('bhal,bhk->bkal', weight, context['variance'])
    selected_age = torch.einsum('bhal,bh->bal', weight, age)
    valid = (mass > 1e-08) & mask[..., None]
    movement = context['now_path'][:, :, None, None] - past_path
    expected = torch.einsum('bkqaz,bkalz->bkqal', gradient, movement)
    raw = rssi[:, None, None, :, None] - past_rssi[:, None, None]
    corrected = raw - delta[:, :, None, :, None] + past_delta[:, :, None]
    residual = corrected - expected
    old_axis = context['axes']
    now_axis = context['now_axis'][:, None]
    cosine = (old_axis * now_axis).sum(-1) / (old_axis.norm(dim=-1) * now_axis.norm(dim=-1)).clamp_min(1e-06)
    cosine = torch.einsum('bhal,bh->bal', weight, cosine)
    uncertainty = (context['now_variance'][..., None, None] + past_variance).clamp_min(1e-06).sqrt()
    shape = expected.shape
    channels = [raw.clamp(-30, 30) / 10, corrected.clamp(-30, 30) / 10, expected.clamp(-30, 30) / 10, residual.clamp(-30, 30) / 10, selected_age[:, None, None] / 2, cosine[:, None, None], movement.norm(dim=-1)[:, :, None].clamp_max(10) / 2, uncertainty[:, :, None] / 4, valid[:, None, None].float()]
    ap = torch.stack([value.expand(shape) for value in channels], -1) * valid[:, None, None, ..., None]
    ap = ap.reshape(b, k, q, a, 9 * len(lags))
    left, right = torch.triu_indices(a, a, 1, device=gradient.device)
    pair_mask = (valid[..., left, :] & valid[..., right, :])[:, None, None]
    raw = raw.expand(shape)
    pair = torch.stack(((raw[..., left, :] - raw[..., right, :]).clamp(-30, 30) / 10, (expected[..., left, :] - expected[..., right, :]).clamp(-30, 30) / 10, (residual[..., left, :] - residual[..., right, :]).clamp(-30, 30) / 10, pair_mask.expand(b, k, q, len(left), len(lags)).float()), -1)
    pair = pair * pair_mask[..., None]
    return (ap, pair.reshape(b, k, q, len(left), 4 * len(lags)))

def upgrade_head(net, count):
    from base_model import SpatialRSSIHead
    if net.temporal_contrast:
        raise ValueError('An existing temporal head needs explicit column migration')
    updated = SpatialRSSIHead(net.variant, net.localization.history_context, net.geometry_head, count)
    state = updated.state_dict()
    for name, value in net.localization.state_dict().items():
        if state[name].shape == value.shape:
            state[name] = value
        elif name in ('ap.0.weight', 'pair.0.weight'):
            state[name].zero_()
            state[name][:, :value.shape[1]] = value
        else:
            raise ValueError('Unexpected temporal-head parameter: ' + name)
    updated.load_state_dict(state, strict=True)
    net.localization = updated
    net.temporal_contrast = count
