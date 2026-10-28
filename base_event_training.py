from collections import defaultdict
import math
import numpy as np
import torch
from base_radio_likelihood import robust_radio_score

class EventSamples:

    def __init__(self, entries):
        self.entries = entries
        self.by_device = defaultdict(list)
        labels = 0
        for index, entry in enumerate(entries):
            event = entry.get('cs_events')
            if event is None:
                continue
            if 'motion_history' not in entry or 'native_innovations' not in entry:
                raise ValueError('CS events require exact native Wi-Fi histories and innovations')
            for row, wifi in enumerate(event['wifi_index']):
                if int(wifi) >= 5:
                    self.by_device[entry['name']].append((index, row))
                    labels += int(event['label_mask'][row])
        self.phones = list(self.by_device)
        self.summary = dict(CS_event_samples=sum(map(len, self.by_device.values())), CS_event_labels=labels, CS_event_phones=self.phones, CS_targets='Source event-ID labels, with pre-CS bias uncertainty', feedback_direction='Earlier arrived Wi-Fi to later CS; no future Wi-Fi')

    @staticmethod
    def wifi_context(entry, index, history):
        lo = max(0, index - history)
        pad = history - (index - lo)
        radio = np.pad(entry['radio'][lo:index], ((pad, 0), (0, 0), (0, 0))).transpose(1, 0, 2)
        innovation = np.pad(entry['native_innovations'][lo:index], ((pad, 0), (0, 0), (0, 0))).transpose(1, 2, 0)
        ages = np.pad(entry['times'][index] - entry['times'][lo:index], (pad, 0), constant_values=1000.0)
        previous = np.trace(entry['covariance'][lo:index], axis1=-2, axis2=-1)
        current = np.trace(entry['covariance'][index], axis1=-2, axis2=-1)
        distance = ((entry['means'][lo:index] - entry['means'][index]) ** 2).sum(-1)
        spatial = np.pad(np.exp(-0.5 * distance / (1 + previous + current[None])), ((pad, 0), (0, 0))).T
        axis = entry['motion'][lo:index][:, [0, 2]]
        now = entry['motion'][index, [0, 2]]
        cosine = (axis * now).sum(-1) / (np.linalg.norm(axis, axis=-1) * np.linalg.norm(now)).clip(1e-06)
        rotation = np.pad(np.exp(2 * (cosine.clip(-1, 1) - 1)), (pad, 0))
        return dict(rssi=entry['rssi'][index], mask=entry['mask'][index], radio_history=radio, initial_history=entry['radio'][:5].transpose(1, 0, 2), motion_history=entry['motion_history'][index], native_innovation_history=innovation, age_history=ages, spatial_history=spatial, rotation_history=rotation, means=entry['means'][index], covariance=entry['covariance'][index], headings=entry['headings'][index], motion=entry['motion'][index], cs_age=entry['cs_age'][index], device_mean=entry['prefix_device_mean'][index], device_covariance=entry['prefix_device_covariance'][index])

    def sample(self, rng, count, history=16):
        output = defaultdict(list)
        fields = ['features', 'motion_history', 'range_m', 'bias_mean', 'bias_variance', 'mode_weights', 'velocity', 'anchor_index', 'wifi_age_s', 'target_range', 'label_mask']
        for _ in range(count):
            phone = self.phones[int(rng.integers(len(self.phones)))]
            items = self.by_device[phone]
            index, row = items[int(rng.integers(len(items)))]
            entry = self.entries[index]
            event = entry['cs_events']
            for key in fields:
                output['event_' + key].append(event[key][row])
            context = self.wifi_context(entry, int(event['wifi_index'][row]), history)
            for key, value in context.items():
                output['event_wifi_' + key].append(value)
        return {key: np.asarray(value) for key, value in output.items()}

def event_objectives(net, batch, grid_fields, anchors, observation_score, field_expand, short_forecast):
    feature = batch['event_features']
    n = len(feature)
    device = feature.device
    event_motion = torch.tanh(net.calibration.motion_encoder(batch['event_motion_history'])[..., -1, :])
    sigma, pi = net.calibration.cs_parameters(feature, event_motion)
    ranges = batch['event_range_m']
    bias = batch['event_bias_mean']
    bias_var = batch['event_bias_variance']
    k = bias.shape[1]
    prior = batch['event_mode_weights'].clamp_min(1e-20).log()
    error = ranges[:, None] - batch['event_target_range'][:, None] - bias
    variance = sigma[:, None].square() + 0.15 ** 2 + bias_var.clamp_min(0.0)
    nominal = -0.5 * (error.square() / variance + variance.log() + math.log(2 * math.pi))
    constant = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi) - math.log(10.0)
    tail = constant - 2 * torch.log1p(error.square() / 300)
    probability = torch.logaddexp(torch.log1p(-pi[:, None]) + nominal, pi[:, None].log() + tail)
    labels = batch['event_label_mask']
    cs_nll = -(torch.logsumexp(prior + probability, -1) * labels).sum() / labels.sum().clamp_min(1)
    prefix = 'event_wifi_'
    fb = {key[len(prefix):]: value for key, value in batch.items() if key.startswith(prefix)}
    a = fb['rssi'].shape[-1]
    context = net.calibration.encode_history(fb['radio_history'], fb['motion_history'], fb['initial_history'], fb['mask'])
    packed = {key: value[:, None, None] for key, value in fb.items()}
    delta = short_forecast(net, packed, context)
    component = torch.randint(k, (n,), device=device)
    arange = torch.arange(n, device=device)
    mu = fb['device_mean'][arange, component]
    cov = fb['device_covariance'][arange, component]
    delta = delta[arange, component]
    g = len(net.field.xy)
    points = net.field.xy[None].expand(n, g, 2)
    field = net.field(points, context['motion'][:, None].expand(n, g, 32), field_expand(grid_fields, (n,)), heading=fb['headings'][arange, component, None])
    basis = field['basis']
    mean = field['mean'] + (basis @ mu[:, None, :, None]).squeeze(-1) + delta[:, None]
    noise = field['map_variance'] + context['sigma'][:, None].square()
    y = fb['rssi'][:, None]
    mask = fb['mask'][:, None].expand(n, g, a)
    ll, s, chol = observation_score(y, mask, mean, noise, basis, cov[:, None])
    dv = (basis @ cov[:, None] @ basis.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1)
    score = ll + net.localization(y, mask, mean, s.diagonal(dim1=-2, dim2=-1), field, dv, net.field.ap_embedding.weight, context['spatial_radio'][:, None].expand(n, g, a, 24), context['spatial_motion'][:, None].expand(n, g, 32), s, chol)
    if net.robust_radio:
        score, _ = robust_radio_score(score, y, mask, mean, s, context['outlier'][:, None])
    age = batch['event_wifi_age_s']
    shifted = points + batch['event_velocity'][arange, component, None] * age[:, None, None]
    anchor = anchors[batch['event_anchor_index'].long()]
    predicted = (shifted - anchor[:, None]).square().sum(-1).clamp_min(1e-08).sqrt() + bias[arange, component, None]
    error = ranges[:, None] - predicted
    variance = sigma.square() + 0.15 ** 2 + 4 * age.clamp_min(0.0).pow(3) / 3 + bias_var[arange, component].clamp_min(0.0)
    nominal = -0.5 * (error.square() / variance[:, None] + variance[:, None].log() + math.log(2 * math.pi))
    tail = constant - 2 * torch.log1p(error.square() / 300)
    likelihood = torch.logaddexp(torch.log1p(-pi[:, None]) + nominal, pi[:, None].log() + tail)
    prediction = torch.logsumexp(score.log_softmax(-1) + likelihood, -1)
    usable = (age >= 0) & (age <= 1.5)
    feedback = -(prediction * usable).sum() / usable.sum().clamp_min(1)
    return (cs_nll, feedback, sigma, pi, labels.sum(), usable.sum())
