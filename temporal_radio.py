import copy
import math
from pathlib import Path
import torch
from torch import nn
from base_model import mlp
from base_evaluate import load_actor as ORIGINAL_LOAD_ACTOR
KIND = 'causal_source_difference_residual'
LAGS = (0.35, 0.9, 1.8)
PARENT_RELATIVE = 'runs/counterfactual_v79_set/initial_window_evidence_phase/epoch_291.pt'
PARENT_SHA = '5e1fc5ddf62c983523283a6eb901ca4180297c9060ffa291740cc31664636e2b'

def features(points, field, correction, rssi, mask, context):
    b, k, q, a = field['mean'].shape
    age = -context['times']
    lag = age.new_tensor(LAGS)
    time_weight = torch.exp(-0.5 * ((age[..., None] - lag) / (0.1 + 0.2 * lag)).square())
    time_weight *= ((age > 0) & (age <= 3.0))[..., None]
    weight = time_weight[:, :, None] * context['mask'][..., None]
    mass = weight.sum(1)
    weight = weight / mass.clamp_min(1e-08)[:, None]
    old_rssi = torch.einsum('bhal,bha->bal', weight, context['rssi'])
    old_source = torch.einsum('bhal,bhka->bkal', weight, context['source'])
    old_correction = torch.einsum('bhal,bhka->bkal', weight, context['correction'])
    old_position = torch.einsum('bhal,bhkz->bkalz', weight, context['position'])
    old_variance = torch.einsum('bhal,bhk->bkal', weight, context['variance'])
    elapsed = torch.einsum('bhal,bh->bal', weight, age)
    valid = (mass > 1e-08) & mask[..., None]
    shape = (b, k, q, a, len(LAGS))
    raw = (rssi[:, None, None, :, None] - old_rssi[:, None, None]).expand(shape)
    calibrated = raw - correction[..., None] + old_correction[:, :, None]
    expected = field['mean'][..., None] - old_source[:, :, None]
    residual = calibrated - expected
    displacement = points[..., None, None, :] - old_position[:, :, None]
    channels = (raw.clamp(-40, 40) / 10, calibrated.clamp(-40, 40) / 10, expected.clamp(-40, 40) / 10, residual.clamp(-40, 40) / 10, displacement[..., 0].clamp(-15, 15) / 3, displacement[..., 1].clamp(-15, 15) / 3, elapsed[:, None, None] / 2, old_variance[:, :, None].clamp_min(0).sqrt() / 4, valid[:, None, None].float())
    ap = torch.stack([v.expand(shape) for v in channels], -1)
    ap *= valid[:, None, None, ..., None]
    left, right = torch.triu_indices(a, a, 1, device=points.device)
    pair_valid = (valid[:, left] & valid[:, right])[:, None, None]
    pair = torch.stack(((calibrated[..., left, :] - calibrated[..., right, :]).clamp(-40, 40) / 10, (expected[..., left, :] - expected[..., right, :]).clamp(-40, 40) / 10, (residual[..., left, :] - residual[..., right, :]).clamp(-40, 40) / 10, pair_valid.expand(b, k, q, len(left), len(LAGS)).float()), -1)
    pair *= pair_valid[..., None]
    return (ap.reshape(b, k, q, a, 27), pair.reshape(b, k, q, len(left), 12))

class TemporalRadioResidual(nn.Module):

    def __init__(self):
        super().__init__()
        self.ap = mlp(43, 48, 24)
        self.ap_out = mlp(48, 32, 1, last_zero=True)
        self.pair = mlp(33, 32, 16)
        self.pair_out = mlp(32, 16, 1, last_zero=True)

    @staticmethod
    def pool(hidden, valid):
        weight = valid[..., None].float()
        count = weight.sum(-2).clamp_min(1.0)
        mean = (hidden * weight).sum(-2) / count
        variance = ((hidden - mean[..., None, :]).square() * weight).sum(-2) / count
        return torch.cat((mean, (variance + 1e-06).sqrt()), -1)

    def forward(self, ap, pair, mask, pair_mask):
        av = self.ap_out(self.pool(self.ap(ap), mask)).squeeze(-1)
        pv = self.pair_out(self.pool(self.pair(pair), pair_mask)).squeeze(-1)
        return 4 * ((av + pv).float() / 4).tanh()

def head_inputs(rssi, mask, prediction, variance, field, device_variance, embedding, temporal):
    error = (rssi - prediction) / variance.clamp_min(0.25).sqrt()
    scalar = torch.stack((error, variance.clamp_min(0.25).log() / 5, field['mean'] / 50, field['gradient'][..., 0] / 10, field['gradient'][..., 1] / 10, device_variance / 100, field['visibility'], mask.float()), -1)
    ap_embedding = embedding.expand(*scalar.shape[:-1], 8)
    ap = torch.cat((scalar, ap_embedding, temporal[0]), -1)
    a = prediction.shape[-1]
    left, right = torch.triu_indices(a, a, 1, device=prediction.device)
    pair_mask = mask[..., left] & mask[..., right]
    source = field['mean'][..., left] - field['mean'][..., right]
    residual = (rssi[..., left] - rssi[..., right] - (prediction[..., left] - prediction[..., right])) / 10
    gradient = field['gradient'][..., left, :] - field['gradient'][..., right, :]
    pair_scalar = torch.cat((residual[..., None], (source / 20)[..., None], gradient / 10, pair_mask[..., None].float()), -1)
    pair = torch.cat((pair_scalar, ap_embedding[..., left, :], ap_embedding[..., right, :], temporal[1]), -1)
    valid = mask & (temporal[0][..., [8, 17, 26]].sum(-1) > 0)
    pair_valid = pair_mask & (temporal[1][..., [3, 7, 11]].sum(-1) > 0)
    return (ap, pair, valid, pair_valid)

class SpatialTemporalHead(nn.Module):

    def __init__(self, static):
        super().__init__()
        self.static = static
        self.residual = TemporalRadioResidual()
        self.history_context = static.history_context
        self.temporal_contrast = 0
        self.last_residual = None

    def forward(self, rssi, mask, prediction, variance, field, device_variance, ap_embedding, evidence=None, motion=None, observation_covariance=None, observation_cholesky=None, temporal=None):
        static = self.static(rssi, mask, prediction, variance, field, device_variance, ap_embedding, evidence, motion, observation_covariance, observation_cholesky)
        if temporal is None:
            raise ValueError('Radio displacement head needs strictly causal context')
        ap, pair, valid, pair_valid = head_inputs(rssi, mask, prediction, variance, field, device_variance, ap_embedding, temporal)
        value = self.residual(ap, pair, valid, pair_valid)
        present = valid.any(-1)
        value = value * present
        self.last_residual = value.detach()
        return static + value

def upgrade(net):
    if net.temporal_contrast or getattr(net, 'direction_temporal_head', None):
        raise ValueError('Expected the selected static epoch291 architecture')
    net.localization = SpatialTemporalHead(net.localization)
    net.temporal_contrast = 3
    net.direction_temporal_head = KIND
    net.requires_grad_(False)
    net.localization.residual.requires_grad_(True)
    return net

def load_actor(base, checkpoint):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    temporal = saved['architecture'].get('direction_temporal_head') == KIND
    speed = bool(saved['architecture'].get('direction_speed_model'))
    if not temporal and (not speed):
        return ORIGINAL_LOAD_ACTOR(base, checkpoint)
    parent = Path(base) / PARENT_RELATIVE
    if saved['parent_checkpoint_sha256'] != PARENT_SHA:
        raise ValueError('Wrong selected parent')
    net, cfg, aps, data, _ = ORIGINAL_LOAD_ACTOR(base, parent)
    if temporal:
        upgrade(net)
    if speed:
        from speed_magnitude import SpeedHead
        net.motion_speed_head = SpeedHead()
    net.load_state_dict(saved['state_dict'], strict=True)
    net.eval()
    return (net, cfg, aps, data, saved)

def checkpoint_metadata(saved, net, policy, updates):
    result = copy.deepcopy(saved)
    result['state_dict'] = {name: value.detach().cpu() for name, value in net.state_dict().items()}
    result['parent_checkpoint_sha256'] = PARENT_SHA
    result['architecture'] = dict(saved['architecture'], direction_temporal_head=KIND, temporal_contrast=3, temporal_lags_s=list(LAGS), direction_policy=policy, temporal_AP_MLP=[43, 48, 24], temporal_AP_pool=[48, 32, 1], temporal_pair_MLP=[33, 32, 16], temporal_pair_pool=[32, 16, 1])
    result['source_temporal_updates'] = int(updates)
    result['training_devices'] = ['oppo2', 'oppo3', 'maprealme', 'vivo']
    result['direction_training_protocol'] = 'Source-only spatial NLL, causal self-predicted history; frozen M1/M2/static M3'
    result['neural_weights_changed'] = True
    return result
