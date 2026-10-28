from __future__ import annotations
import argparse
import gc
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
import sys
from pathlib import Path
import time
import numpy as np
if int(np.__version__.split('.')[0]) < 2:
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
import torch
from torch.nn import functional as F
from base_data import BootstrapEpisodes
from base_model import ThreeModuleNetwork, gaussian_log_prob, stabilize
from base_quadrature import position_support
from base_radio_likelihood import robust_radio_score
_DATA = None
AVAILABLE_CPUS = set(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def sample_chunk(arguments):
    seed, batch = arguments
    torch.set_num_threads(1)
    if AVAILABLE_CPUS:
        os.sched_setaffinity(0, AVAILABLE_CPUS)
    return {key: value.numpy() for key, value in _DATA.sample(np.random.default_rng(seed), batch=batch, device='cpu').items()}

def enqueue(pool, seed, batch, workers):
    counts = [batch // workers + int(i < batch % workers) for i in range(workers)]
    return [pool.submit(sample_chunk, (seed + i, count)) for i, count in enumerate(counts) if count]

def peer_requires_training(state_path):
    state = json.loads(state_path.read_text())
    peer = next((row for row in state['jobs'] if row['pid'] != os.getpid()))
    process = Path('/proc') / str(peer['pid'])
    if not process.exists():
        return False
    try:
        if (process / 'stat').read_text().split(') ', 1)[1].startswith('Z'):
            return False
    except FileNotFoundError:
        return False
    marker = Path(peer['root']) / 'nominal_training_complete.json'
    if marker.exists():
        finished = json.loads(marker.read_text())
        if finished['pid'] == peer['pid']:
            return False
    return True

def perform_update(net, tensors, grid_fields, anchors, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = losses(net, tensors, grid_fields, anchors)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite actual source objective')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    optimizer.step()
    row = {k: float(v) for k, v in metrics.items()}
    row.update(gradient_norm=float(norm), batch=len(tensors['rssi']), source_frames=int(tensors['valid'].sum()), GPU_peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2 ** 30, GPU_reserved_GiB=torch.cuda.memory_reserved() / 2 ** 30)
    if net.localization.history_context:
        row['M3_history_weight_norm'] = float(net.localization.ap[0].weight[:, 16:].detach().norm())
    gain = getattr(net.localization, 'last_gain', None)
    if gain is not None:
        row.update(spatial_precision_gain_mean=float(gain.mean()), spatial_precision_gain_p95=float(torch.quantile(gain.reshape(-1), 0.95)))
    if net.orientation_basis:
        row['orientation_modulation_mean'] = float(net.field.orientation_strength.detach().tanh().abs().mean())
    if net.ap_basis_prior:
        row['AP_basis_prior_strength'] = float(net.field.AP_strength().detach())
    if net.initial_posterior != 'none':
        row['initial_device_shift_norm'] = float(net.calibration.initial_encoder.last_shift.norm(dim=-1).mean())
        row['initial_device_scale_mean'] = float(net.calibration.initial_encoder.last_scale.mean())
    return row

def field_expand(fields, shape):
    result = {}
    for key, value in fields.items():
        result[key] = value.expand(*shape, *value.shape)
    return result

def observation_score(y, mask, prediction, noise, basis, device_covariance):
    covariance = basis @ device_covariance @ basis.transpose(-1, -2) + torch.diag_embed(noise)
    m = mask.float()
    covariance = covariance * m[..., None] * m[..., None, :] + torch.diag_embed(1 - m)
    score, chol = gaussian_log_prob((y - prediction) * m, stabilize(covariance), return_factor=True)
    return (score + 0.5 * (1 - m).sum(-1) * np.log(2 * np.pi), covariance, chol)

def native_short_forecast(net, batch, context):
    b, r, t, a = batch['rssi'].shape
    flat = b * r * t
    k = batch['means'].shape[-2]
    history = batch['radio_history'].reshape(flat, a, -1, 8)
    innovation = batch['native_innovation_history'].reshape(flat, k, a, -1)
    available = history[..., 5].bool()[:, None].expand(flat, k, a, history.shape[-2])
    ages = batch['age_history'].reshape(flat, -1)[:, None].expand(flat, k, -1)
    spatial = batch['spatial_history'].reshape(flat, k, -1)
    confidence = (1 + batch['covariance'].diagonal(dim1=-2, dim2=-1).sum(-1) / 4).rsqrt().reshape(flat, k)
    gap = batch['motion'].reshape(flat, 41)[:, 20] * 5
    cs_presence = (batch['cs_age'] <= 1.5).float().mean(-1).reshape(flat)
    delta, _ = net.calibration.short_residual(context, net.field.ap_embedding.weight, innovation, ages, available, confidence, spatial, batch['rotation_history'].reshape(flat, 1, -1).expand_as(spatial), gap, cs_presence)
    return delta

def source_initial_cavity(net, batch):
    b, a = batch['initial_rssi'].shape
    k = batch['first_means'].shape[1]
    device = batch['rssi'].device
    if 'prefix_region_information' not in batch:
        raise ValueError('Initial recognition needs its native canonical prefix')
    if net.initial_recognition != 'first':
        from base_initial_posterior import window_posterior
        if 'startup_poses' not in batch:
            raise ValueError('Initial window needs matching causal source startup captures')
        scans = batch['startup_rssi'].shape[1]
        context = net.calibration.encode_history(batch['startup_radio_history'].reshape(b * scans, a, 16, 8), batch['startup_motion_history'].reshape(b * scans, 16, 41), batch['startup_initial_history'].reshape(b * scans, a, 5, 8), batch['startup_mask'].reshape(b * scans, a))
        motion = context['motion'].reshape(b, scans, 32)[:, :, None].expand(b, scans, k, 32)
        field = net.field(batch['startup_poses'], motion, heading=batch['startup_headings'])
        if net.calibrated_geometry:
            from base_spatial_geometry import calibrated_field
            field = calibrated_field(field, batch['startup_device_mean'])
        noise = torch.diag_embed(field['map_variance'] + context['sigma'].reshape(b, scans, 1, a).square()) + field['gradient'] @ stabilize(batch['startup_pose_covariance']) @ field['gradient'].transpose(-1, -2)
        if net.robust_radio:
            noise = noise / batch['startup_nominal_responsibility'].clamp_min(0.001)[..., None, None]
        else:
            scale = (1 - context['outlier'].reshape(b, scans, 1, a)).clamp_min(0.05).rsqrt()
            noise = noise * scale[..., None] * scale[..., None, :]
        corrected = batch['startup_rssi'][:, :, None] - batch['startup_delta']
        mask = batch['startup_mask'][:, :, None].expand(b, scans, k, a)
        window_field = {key: field[key].transpose(1, 2) for key in ['mean', 'basis', 'gradient']}
        _, _, (mean, cov) = window_posterior(net.calibration, corrected.transpose(1, 2), mask.transpose(1, 2), window_field, noise.transpose(1, 2), net.initial_recognition, net.field.ap_embedding.weight)
        net.calibration.current_initial_base = (mean, cov)
        result = net.calibration.posterior(batch['prefix_region_information'], batch['prefix_region_vector'], mean, cov)
        net.calibration.last_initial_prefix_mean = result[0].detach()
        net.calibration.last_initial_prefix_covariance = result[1].detach()
        return result
    context = net.calibration.encode_history(torch.zeros(b, a, 16, 8, device=device), batch['first_motion_history'], torch.zeros(b, a, 5, 8, device=device), batch['initial_mask'])
    field = net.field(batch['first_means'], context['motion'][:, None].expand(b, k, 32), heading=batch['first_headings'])
    if net.calibrated_geometry:
        from base_spatial_geometry import calibrated_field
        prior_mean, _ = net.calibration.prior((b, k))
        field = calibrated_field(field, prior_mean)
    noise = torch.diag_embed(field['map_variance'] + context['sigma'][:, None].square()) + field['gradient'] @ stabilize(batch['first_covariance']) @ field['gradient'].transpose(-1, -2)
    if net.robust_radio:
        noise = noise / batch['first_nominal_responsibility'].clamp_min(0.001)[..., None, None]
    else:
        scale = (1 - context['outlier'][:, None]).clamp_min(0.05).rsqrt()
        noise = noise * scale[..., None] * scale[..., None, :]
    information, vector, _, _, _ = net.calibration.block_information(field['basis'][:, :, None], (batch['initial_rssi'][:, None] - field['mean'])[:, :, None], noise[:, :, None], batch['initial_mask'][:, None, None].expand(b, k, 1, a))
    mean, cov = net.calibration.posterior(information, vector)
    mean, cov = net.calibration.initial_encoder(batch['initial_rssi'][:, None], batch['initial_mask'][:, None], field, noise.diagonal(dim1=-2, dim2=-1), mean, cov, net.field.ap_embedding.weight)
    net.calibration.current_initial_base = (mean, cov)
    result = net.calibration.posterior(batch['prefix_region_information'], batch['prefix_region_vector'], mean, cov)
    net.calibration.last_initial_prefix_mean = result[0].detach()
    net.calibration.last_initial_prefix_covariance = result[1].detach()
    return result

def joint_mode_position(net, batch, context, shared_field, mu, cov, delta, target, position_rule='mixed', transfer_score=None):
    b, r, t, a = batch['rssi'].shape
    flat = b * r * t
    k = mu.shape[1]
    prefix = batch['means'].reshape(flat, k, 2)
    covariance = stabilize(batch['covariance'].reshape(flat, k, 2, 2))
    grid, prior, g = position_support(prefix, covariance, net.field.xy, position_rule)
    n = grid.shape[-2]
    count = flat * k
    local_motion = context['motion'][:, None, None].expand(flat, k, n - g, 32).reshape(count, n - g, 32)
    local_field = net.field(grid[:, :, g:].reshape(count, n - g, 2), local_motion)
    field = {key: torch.cat((value[:, None].expand(flat, k, *value.shape[1:]).reshape(count, *value.shape[1:]), local_field[key]), 1) for key, value in shared_field.items()}
    heading = batch['headings'].reshape(count)
    field = net.field.condition_heading(field, grid.reshape(count, n, 2), heading[:, None])
    basis = field['basis']
    cm = mu.reshape(count, 6)
    cc = cov.reshape(count, 6, 6)
    prediction = field['mean'] + (basis @ cm[:, None, :, None]).squeeze(-1) + delta.reshape(count, 1, a)
    sigma = context['sigma'][:, None, None].expand(flat, k, n, a).reshape(count, n, a)
    y = batch['rssi'].reshape(flat, a)[:, None, None].expand(flat, k, n, a).reshape(count, n, a)
    mask = batch['mask'].reshape(flat, a)[:, None, None].expand(flat, k, n, a).reshape(count, n, a)
    noise = field['map_variance'] + sigma.square()
    ll, s, radio_chol = observation_score(y, mask, prediction, noise, basis, cc[:, None])
    dv = (basis @ cc[:, None] @ basis.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1)
    radio = context['spatial_radio'][:, None, None].expand(flat, k, n, a, 24).reshape(count, n, a, 24) if net.localization.history_context else None
    head_motion = context['spatial_motion'][:, None, None].expand(flat, k, n, 32).reshape(count, n, 32)
    score = ll + net.localization(y, mask, prediction, s.diagonal(dim1=-2, dim2=-1), field, dv, net.field.ap_embedding.weight, radio, head_motion, s, radio_chol)
    responsibility = None
    if net.robust_radio:
        pi = context['outlier'][:, None, None].expand(flat, k, n, a).reshape(count, n, a)
        score, responsibility = robust_radio_score(score, y, mask, prediction, s, pi)
    score = score.reshape(flat, k, n)
    evidence = torch.logsumexp(prior + score, -1)
    previous = batch['weights'].reshape(flat, k).clamp_min(1e-20).log()
    log_joint = previous[..., None] + prior + score
    log_z = torch.logsumexp(log_joint.reshape(flat, -1), -1)
    posterior = (log_joint - log_z[:, None, None]).exp()
    branch_log_weight = (previous + evidence).log_softmax(-1)
    if transfer_score is not None:
        branch_log_weight = (previous + transfer_score).log_softmax(-1)
        posterior = branch_log_weight.exp()[..., None] * (prior + score - evidence[..., None]).exp()
    position = (posterior[..., None] * grid).sum((1, 2))
    valid = batch['valid'].reshape(flat)
    error = (position - target).square().sum(-1).clamp_min(1e-08).sqrt()
    risk = (F.smooth_l1_loss(position, target, reduction='none').sum(-1) * valid).sum() / valid.sum().clamp_min(1)
    tail = error[valid].topk(max(1, int(valid.sum().item() * 0.1))).values.mean()
    truth_xy = target[:, None].expand(flat, k, 2).reshape(count, 2)
    truth = net.field(truth_xy, context['motion'][:, None].expand(flat, k, 32).reshape(count, 32), heading=heading)
    truth_prediction = truth['mean'] + (truth['basis'] @ cm[..., None]).squeeze(-1) + delta.reshape(count, a)
    truth_y = batch['rssi'].reshape(flat, a)[:, None].expand(flat, k, a).reshape(count, a)
    truth_mask = batch['mask'].reshape(flat, a)[:, None].expand(flat, k, a).reshape(count, a)
    truth_sigma = context['sigma'][:, None].expand(flat, k, a).reshape(count, a)
    truth_ll, truth_s, truth_chol = observation_score(truth_y, truth_mask, truth_prediction, truth['map_variance'] + truth_sigma.square(), truth['basis'], cc)
    truth_dv = (truth['basis'] @ cc @ truth['basis'].transpose(-1, -2)).diagonal(dim1=-2, dim2=-1)
    truth_score = truth_ll + net.localization(truth_y, truth_mask, truth_prediction, truth_s.diagonal(dim1=-2, dim2=-1), truth, truth_dv, net.field.ap_embedding.weight, context['spatial_radio'][:, None].expand(flat, k, a, 24).reshape(count, a, 24) if net.localization.history_context else None, context['spatial_motion'][:, None].expand(flat, k, 32).reshape(count, 32), truth_s, truth_chol)
    if net.robust_radio:
        pi = context['outlier'][:, None].expand(flat, k, a).reshape(count, a)
        truth_score, _ = robust_radio_score(truth_score, truth_y, truth_mask, truth_prediction, truth_s, pi)
    truth_prior = gaussian_log_prob(target[:, None] - prefix, covariance)
    density = torch.logsumexp(previous + truth_score.reshape(flat, k) + truth_prior, -1) - log_z
    if transfer_score is not None:
        density = torch.logsumexp(branch_log_weight + truth_score.reshape(flat, k) + truth_prior - evidence, -1)
    nll = -(density * valid).sum() / valid.sum().clamp_min(1)
    branch_weight = branch_log_weight.exp()
    entropy = -(branch_weight * branch_weight.clamp_min(1e-20).log()).sum(-1).mean()
    radio_logp = torch.logsumexp(previous + truth_score.reshape(flat, k), -1)
    observed = batch['mask'].reshape(flat, a).sum(-1).float()
    radio_nll = -(radio_logp * valid).sum() / (observed * valid).sum().clamp_min(1)
    nominal_mass = (posterior * responsibility.reshape(flat, k, n)).sum((1, 2)) if responsibility is not None else torch.ones(flat, device=target.device)
    return (risk, nll, tail, error[valid].median(), entropy, radio_nll, nominal_mass[valid].mean())

def losses(net, batch, grid_fields, anchors, path_mixture=False, continuous_position=False, joint_modes=False, position_rule='mixed', defer_cs_supervision=False, device_region_weight=0.3):
    rssi, mask = (batch['rssi'], batch['mask'])
    b, r, t, a = rssi.shape
    k = batch['means'].shape[-2]
    flat = b * r * t
    context = net.calibration.encode_history(batch['radio_history'].reshape(flat, a, -1, 8), batch['motion_history'].reshape(flat, -1, 41), batch['initial_history'].reshape(flat, a, 5, 8), mask.reshape(flat, a))
    motion = context['motion'].reshape(b, r, t, 32)
    points = batch['means'].transpose(2, 3)
    mc = motion[:, :, None].expand(b, r, k, t, 32)
    geometry = net.field(points, mc, heading=batch['headings'].transpose(2, 3))
    variance = context['sigma'].square().reshape(b, r, 1, t, a) + geometry['map_variance']
    pose_variance = torch.einsum('brktad,brktde,brktae->brkta', geometry['gradient'], batch['covariance'].transpose(2, 3), geometry['gradient'])
    variance = variance + pose_variance.clamp_min(0.0)
    observation_weight = batch['radio_nominal_responsibility'].transpose(2, 3)[..., None].clamp_min(0.001) if net.robust_radio else (1 - context['outlier'].reshape(b, r, 1, t, a)).clamp_min(0.05)
    variance = variance / observation_weight
    block_covariance = None
    if net.full_pose_covariance:
        pose_block = geometry['gradient'] @ batch['covariance'].transpose(2, 3) @ geometry['gradient'].transpose(-1, -2)
        radio_variance = context['sigma'].square().reshape(b, r, 1, t, a) + geometry['map_variance']
        block_covariance = pose_block + torch.diag_embed(radio_variance)
        precision_weight = observation_weight.rsqrt()
        block_covariance = block_covariance * precision_weight[..., None] * precision_weight[..., None, :]
    initial = net.field(batch['initial_xy'].float(), motion[:, 0, 0])
    initial_weight = batch['initial_mask'].float() / (initial['map_variance'] + 9.0)
    initial_information = torch.einsum('bad,ba,bae->bde', initial['basis'], initial_weight, initial['basis'])
    initial_vector = torch.einsum('bad,ba,ba->bd', initial['basis'], initial_weight, batch['initial_rssi'] - initial['mean'])
    causal_delta = native_short_forecast(net, batch, context) if path_mixture and 'native_innovation_history' in batch else None
    geo_prior = batch['weights'][:, :, 0].clamp_min(1e-08).log()
    last = points[:, :, :, -1]
    first = points[:, :, :, 0]
    movement = first[:, 1:, :, None, :] - last[:, :-1, None, :, :]
    transition = (-0.5 * movement.square().sum(-1) / 4).transpose(-1, -2)
    if path_mixture:
        support_offset = None
        forecast_offset = None
        if causal_delta is not None:
            support_offset = causal_delta.reshape(b, r, t, k, a).permute(0, 1, 3, 2, 4)
            forecast_offset = support_offset[:, :, :, :1].expand(b, r, k, t, a).reshape(b, r, k, t * a)
            support_offset = support_offset.reshape(b, r, k, t * a)
        prefix_mean, prefix_covariance = (batch.get('prefix_device_mean'), batch.get('prefix_device_covariance'))
        if net.initial_posterior != 'none':
            prefix_mean, prefix_covariance = source_initial_cavity(net, batch)
        message = net.calibration.path_messages(geometry['mean'].reshape(b, r, k, t * a), geometry['basis'].reshape(b, r, k, t * a, 6), rssi.reshape(b, r, t * a), mask.reshape(b, r, t * a), variance.reshape(b, r, k, t * a), geo_prior, initial_information, initial_vector, prefix_mean, prefix_covariance, support_offset, forecast_offset, block_covariance=block_covariance)
        mu_paths = message['cavity_mean'][:, :, None].expand(b, r, t, k, 6).reshape(flat, k, 6)
        cov_paths = message['cavity_covariance'][:, :, None].expand(b, r, t, k, 6, 6).reshape(flat, k, 6, 6)
        component = torch.multinomial(batch['weights'].reshape(flat, k), 1).squeeze(-1)
        arange = torch.arange(flat, device=rssi.device)
        mu = mu_paths[arange, component]
        cov = cov_paths[arange, component]
    else:
        message = net.calibration.messages(geometry['mean'].reshape(b, r, k, t * a), geometry['basis'].reshape(b, r, k, t * a, 6), rssi.reshape(b, r, t * a), mask.reshape(b, r, t * a), variance.reshape(b, r, k, t * a), geo_prior, transition_log_prob=transition, initial_information=initial_information, initial_vector=initial_vector, causal=True)
        mu = message['cavity_mean'][:, :, None].expand(b, r, t, 6).reshape(flat, 6)
        cov = message['cavity_covariance'][:, :, None].expand(b, r, t, 6, 6).reshape(flat, 6, 6)
    history = batch['radio_history'].reshape(flat, a, -1, 8)
    source = batch['source_history'].reshape(flat, -1, a).transpose(-1, -2)
    memory_mu = mu_paths if path_mixture else mu[:, None]
    memory_modes = memory_mu.shape[1]
    historical_mean = source[:, None] + memory_mu[:, :, 0, None, None] + memory_mu[:, :, 1, None, None] * (source[:, None] / 50)
    innovation = history[:, None, ..., 0] * 100 - historical_mean
    if 'native_innovation_history' in batch:
        innovation = batch['native_innovation_history'].reshape(flat, k, a, -1)
    available = history[..., 5].bool()[:, None].expand(flat, memory_modes, a, history.shape[-2])
    ages = batch['age_history'].reshape(flat, -1)[:, None].expand(flat, memory_modes, -1)
    spatial = batch['spatial_history'].reshape(flat, k, -1)
    if not path_mixture:
        spatial = spatial.mean(1, keepdim=True)
    conf = (1 + batch['covariance'].diagonal(dim1=-2, dim2=-1).sum(-1) / 4).rsqrt().reshape(flat, k)
    if not path_mixture:
        conf = conf.mean(-1, keepdim=True)
    gap = batch['motion'].reshape(flat, 41)[:, 20] * 5
    cs_presence = (batch['cs_age'] <= 1.5).float().mean(-1).reshape(flat)
    if causal_delta is not None:
        delta = causal_delta
    else:
        delta, _ = net.calibration.short_residual(context, net.field.ap_embedding.weight, innovation, ages, available, conf, spatial, batch['rotation_history'].reshape(flat, 1, -1).expand_as(spatial), gap, cs_presence)
    delta_paths = delta
    delta = delta[arange, component] if path_mixture else delta[:, 0]
    target = torch.nan_to_num(batch['targets']).reshape(flat, 2)
    truth = net.field(target, context['motion'])
    if path_mixture:
        truth_basis = truth['basis'][:, None]
        if net.orientation_basis:
            expanded = {key: value[:, None].expand(flat, k, *value.shape[1:]) for key, value in truth.items()}
            truth_basis = net.field.condition_heading(expanded, target[:, None], batch['headings'].reshape(flat, k))['basis']
            truth['basis'] = truth_basis[arange, component]
        component_base = truth['mean'][:, None] + (truth_basis @ mu_paths[..., None]).squeeze(-1)
        component_prediction = component_base + delta_paths
        prediction = (component_prediction * batch['weights'].reshape(flat, k, 1)).sum(1)
        base_prediction = (component_base * batch['weights'].reshape(flat, k, 1)).sum(1)
    else:
        base_prediction = truth['mean'] + (truth['basis'] @ mu[..., None]).squeeze(-1)
        prediction = base_prediction + delta
    valid = (batch['valid'][..., None] & mask).reshape(flat, a)
    error = rssi.reshape(flat, a) - prediction
    mae = (error.abs() * valid).sum() / valid.sum().clamp_min(1)
    base_error = (rssi.reshape(flat, a) - base_prediction).reshape(b, r, t, a)
    region_valid = valid.reshape(b, r, t, a)
    region_count = region_valid.sum(2)
    region_bias = (base_error * region_valid).sum(2) / region_count.clamp_min(1)
    device_region_bias = (region_bias.abs() * (region_count > 0)).sum() / (region_count > 0).sum().clamp_min(1)
    device_only_mae = (base_error.abs() * region_valid).sum() / region_valid.sum().clamp_min(1)
    sigma = context['sigma']
    pi = context['outlier']
    device_uncertainty = torch.einsum('mad,mde,mae->ma', truth['basis'], cov, truth['basis']).clamp_min(0.0)
    rv = truth['map_variance'] + sigma.square() + device_uncertainty
    normal = -0.5 * (error.square() / rv + rv.log() + np.log(2 * np.pi))
    scale = 10.0
    tail = math.lgamma(2) - math.lgamma(1.5) - 0.5 * np.log(3 * np.pi) - np.log(scale) - 2 * torch.log1p(error.square() / (3 * scale ** 2))
    if path_mixture:
        path_var = (truth_basis @ cov_paths @ truth_basis.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1).clamp_min(0.0)
        path_var = path_var + truth['map_variance'][:, None] + sigma[:, None].square()
        path_error = rssi.reshape(flat, 1, a) - component_prediction
        path_normal = -0.5 * (path_error.square() / path_var + path_var.log() + np.log(2 * np.pi))
        path_tail = math.lgamma(2) - math.lgamma(1.5) - 0.5 * np.log(3 * np.pi) - np.log(scale) - 2 * torch.log1p(path_error.square() / (3 * scale ** 2))
        path_logp = (torch.logaddexp((1 - pi[:, None]).log() + path_normal, pi[:, None].log() + path_tail) * valid[:, None]).sum(-1)
        mixture_logp = torch.logsumexp(batch['weights'].reshape(flat, k).clamp_min(1e-20).log() + path_logp, -1)
        observation_nll = -mixture_logp.sum() / valid.sum().clamp_min(1)
    else:
        observation_nll = -(torch.logaddexp((1 - pi).log() + normal, pi.log() + tail) * valid).sum() / valid.sum().clamp_min(1)
    source_grid_count = len(net.field.xy)
    grid = net.field.xy[None].expand(flat, source_grid_count, 2)
    fields = field_expand(grid_fields, (flat,))
    if continuous_position:
        native_mean = batch['means'].reshape(flat, k, 2)[arange, component]
        native_cov = stabilize(batch['covariance'].reshape(flat, k, 2, 2)[arange, component])
        local_chol = torch.linalg.cholesky(native_cov)
        offsets = torch.stack((torch.zeros_like(native_mean), 3 ** 0.5 * local_chol[:, :, 0], -3 ** 0.5 * local_chol[:, :, 0], 3 ** 0.5 * local_chol[:, :, 1], -3 ** 0.5 * local_chol[:, :, 1]), 1)
        local = native_mean[:, None] + offsets
        local_fields = net.field.source(local)
        fields = {key: torch.cat((value, local_fields[key]), 1) for key, value in fields.items()}
        grid = torch.cat((grid, local), 1)
    g = grid.shape[1]
    selected_heading = batch['headings'].reshape(flat, k)[arange, component] if path_mixture else batch['headings'].reshape(flat, k).mean(-1)
    field = net.field(grid, context['motion'][:, None].expand(flat, g, 32), fields, heading=selected_heading[:, None])
    basis = field['basis']
    mean = field['mean'] + (basis @ mu[:, None, :, None]).squeeze(-1) + delta[:, None]
    noise = field['map_variance'] + sigma[:, None].square()
    y = rssi.reshape(flat, a)[:, None]
    m = mask.reshape(flat, a)[:, None].expand(flat, g, a)
    loglike, covariance, radio_chol = observation_score(y, m, mean, noise, basis, cov[:, None])
    device_variance = (basis @ cov[:, None] @ basis.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1)
    correction = net.localization(y, m, mean, covariance.diagonal(dim1=-2, dim2=-1), field, device_variance, net.field.ap_embedding.weight, context['spatial_radio'][:, None].expand(flat, g, a, 24), context['spatial_motion'][:, None].expand(flat, g, 32), covariance, radio_chol)
    wifi_score = loglike + correction
    if net.robust_radio:
        wifi_score, _ = robust_radio_score(wifi_score, y, m, mean, covariance, context['outlier'][:, None])
    wifi_probability = wifi_score.softmax(-1)
    distance = (grid - target[:, None]).square().sum(-1)
    positive = distance.argmin(-1)
    positive_score = wifi_score.gather(-1, positive[:, None])
    if continuous_position:
        truth_prediction = truth['mean'] + (truth['basis'] @ mu[..., None]).squeeze(-1) + delta
        truth_ll, truth_s, truth_chol = observation_score(rssi.reshape(flat, a), mask.reshape(flat, a), truth_prediction, truth['map_variance'] + sigma.square(), truth['basis'], cov)
        truth_score = truth_ll + net.localization(rssi.reshape(flat, a), mask.reshape(flat, a), truth_prediction, truth_s.diagonal(dim1=-2, dim2=-1), truth, device_uncertainty, net.field.ap_embedding.weight, context['spatial_radio'], context['spatial_motion'], truth_s, truth_chol)
        if net.robust_radio:
            truth_score, _ = robust_radio_score(truth_score, rssi.reshape(flat, a), mask.reshape(flat, a), truth_prediction, truth_s, context['outlier'])
        positive_score = truth_score[:, None]
    negatives = wifi_score.masked_fill(distance < 0.6 ** 2, -10000.0).topk(min(16, g), -1).values
    ranking = F.softplus(0.5 + negatives - positive_score).mean(-1)
    if path_mixture:
        geo_mean = batch['means'].reshape(flat, k, 2)[arange, component, None]
        geo_cov = stabilize(batch['covariance'].reshape(flat, k, 2, 2)[arange, component, None] + torch.eye(2, device=y.device) * 0.1)
        geo_weight = torch.zeros(flat, 1, device=y.device)
    else:
        geo_mean = batch['means'].reshape(flat, k, 2)
        geo_cov = stabilize(batch['covariance'].reshape(flat, k, 2, 2) + torch.eye(2, device=y.device) * 0.1)
        geo_weight = batch['weights'].reshape(flat, k).clamp_min(1e-08).log()
    difference = grid[:, None] - geo_mean[:, :, None]
    precision = torch.linalg.inv(geo_cov)
    geo_ll = -0.5 * torch.einsum('bknd,bkde,bkne->bkn', difference, precision, difference)
    geo_ll = geo_ll - 0.5 * torch.linalg.slogdet(geo_cov)[1][..., None] + geo_weight[..., None]
    geo_ll = torch.logsumexp(geo_ll, 1)
    if continuous_position:
        difference_grid = grid[:, :source_grid_count] - native_mean[:, None]
        native_precision = torch.linalg.inv(native_cov)
        global_prior = (-0.5 * torch.einsum('mnd,mde,mne->mn', difference_grid, native_precision, difference_grid)).log_softmax(-1)
        std = native_cov.diagonal(dim1=-2, dim2=-1).mean(-1).sqrt()
        fraction = ((std - 0.15) / 0.45).clamp(0.0, 0.75)
        local_prior = torch.tensor([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6], device=y.device).log().expand(flat, 5)
        geo_ll = torch.cat((global_prior + fraction.clamp_min(1e-20).log()[:, None], local_prior + torch.log1p(-fraction)[:, None]), -1)
    posterior = (wifi_score + geo_ll).softmax(-1)
    position = (posterior[..., None] * grid).sum(-2)
    position_error = (position - target).square().sum(-1).clamp_min(1e-08).sqrt()
    query_valid = batch['valid'].reshape(flat)
    location = F.smooth_l1_loss(position, target, reduction='none').sum(-1)
    location = (location * query_valid).sum() / query_valid.sum().clamp_min(1)
    tail_error = position_error[query_valid].topk(max(1, int(query_valid.sum().item() * 0.1))).values.mean()
    position_nll = -((wifi_score + geo_ll).log_softmax(-1).gather(-1, positive[:, None]).squeeze(-1) * query_valid).sum() / query_valid.sum().clamp_min(1)
    if continuous_position:
        truth_prior = gaussian_log_prob(target - native_mean, native_cov)
        log_normalizer = torch.logsumexp(wifi_score + geo_ll, -1)
        position_nll = -((truth_score + truth_prior - log_normalizer) * query_valid).sum() / query_valid.sum().clamp_min(1)
    mixture_entropy = None
    if joint_modes:
        transfer_score = None
        if net.mode_evidence != 'local':
            from base_transfer_evidence import predictive_transfer
            if 'transfer_mean' not in batch:
                raise ValueError('Regional evidence requires a matching native source prefix')
            transfer_score = predictive_transfer(rssi[:, :, None], mask[:, :, None], geometry, batch['transfer_mean'].transpose(2, 3), batch['transfer_covariance'].transpose(2, 3), batch['transfer_offset'].transpose(2, 3), context['sigma'].reshape(b, r, 1, t, a), context['outlier'].reshape(b, r, 1, t, a), batch['covariance'].transpose(2, 3), robust=net.robust_transfer).transpose(2, 3).reshape(flat, k)
        location, position_nll, tail_error, mixture_median, mixture_entropy, position_radio_nll, nominal_mass = joint_mode_position(net, batch, context, {key: value[:, :source_grid_count] for key, value in field.items()}, mu_paths, cov_paths, delta_paths, target, position_rule, transfer_score)
        if net.precision_head:
            observation_nll = position_radio_nll
    if 'event_features' in batch:
        from base_event_training import event_objectives
        cs_nll, heldout_cs, cs_sigma, cs_pi, event_labels, event_feedback = event_objectives(net, batch, grid_fields, anchors, observation_score, field_expand, native_short_forecast)
    else:
        cs_feature = batch['cs'].reshape(flat, a, 20)
        cs_motion = context['motion'][:, None].expand(flat, a, 32)
        cs_sigma, cs_pi = net.calibration.cs_parameters(cs_feature, cs_motion)
        ranges = batch['ranges'].reshape(flat, a)
        cs_valid = (batch['cs_age'].reshape(flat, a) <= 1.5) & query_valid[:, None]
        true_ranges = (target[:, None] - anchors).square().sum(-1).clamp_min(1e-08).sqrt()
        cs_error = ranges - true_ranges
        cs_normal = -0.5 * (cs_error.square() / cs_sigma.square() + 2 * cs_sigma.log() + np.log(2 * np.pi))
        cs_tail = math.lgamma(2) - math.lgamma(1.5) - 0.5 * np.log(3 * np.pi) - np.log(10.0) - 2 * torch.log1p(cs_error.square() / 300)
        cs_logp = torch.logaddexp((1 - cs_pi).log() + cs_normal, cs_pi.log() + cs_tail)
        cs_nll = -(cs_logp * cs_valid).sum() / cs_valid.sum().clamp_min(1)
        independent_count = source_grid_count if continuous_position else g
        grid_range = (grid[:, :independent_count, None] - anchors).square().sum(-1).clamp_min(1e-08).sqrt()
        range_error = ranges[:, None] - grid_range
        range_variance = cs_sigma[:, None].square() + 0.15 ** 2
        nom = -0.5 * (range_error.square() / range_variance + range_variance.log() + np.log(2 * np.pi))
        tail = math.lgamma(2) - math.lgamma(1.5) - 0.5 * np.log(3 * np.pi) - np.log(10.0) - 2 * torch.log1p(range_error.square() / 300)
        if path_mixture:
            independent = torch.randint(k, (flat,), device=y.device)
            independent_mu = mu_paths[arange, independent]
            independent_cov = cov_paths[arange, independent]
            independent_basis = basis[:, :independent_count]
            independent_field = {key: value[:, :independent_count] for key, value in field.items()}
            independent_mask = m[:, :independent_count]
            independent_mean = independent_field['mean'] + (independent_basis @ independent_mu[:, None, :, None]).squeeze(-1)
            independent_ll, independent_s, independent_chol = observation_score(y, independent_mask, independent_mean, noise[:, :independent_count], independent_basis, independent_cov[:, None])
            independent_var = (independent_basis @ independent_cov[:, None] @ independent_basis.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1)
            independent_score = independent_ll + net.localization(y, independent_mask, independent_mean, independent_s.diagonal(dim1=-2, dim2=-1), independent_field, independent_var, net.field.ap_embedding.weight, context['spatial_radio'][:, None].expand(flat, independent_count, a, 24), context['spatial_motion'][:, None].expand(flat, independent_count, 32), independent_s, independent_chol)
            if net.robust_radio:
                independent_score, _ = robust_radio_score(independent_score, y, independent_mask, independent_mean, independent_s, context['outlier'][:, None])
            wifi_probability = independent_score.softmax(-1)
        predictive = torch.logsumexp(wifi_probability.clamp_min(1e-20).log()[..., None] + torch.logaddexp((1 - cs_pi[:, None]).log() + nom, cs_pi[:, None].log() + tail), 1)
        fresh_cs = cs_valid & (batch['cs_age'].reshape(flat, a) <= gap[:, None].clamp_min(0.0))
        heldout_cs = -(predictive * fresh_cs).sum() / fresh_cs.sum().clamp_min(1)
    region_count = mask.sum((-1, -2)).float()
    region_valid = region_count > 0
    regional_logp = torch.logsumexp(geo_prior.log_softmax(-1) + message['heldout_scores'], -1)
    cross_rssi = -(regional_logp / region_count.clamp_min(1) * region_valid).sum() / region_valid.sum().clamp_min(1)
    rank = (ranking * query_valid).sum() / query_valid.sum().clamp_min(1)
    regularization = 0.005 * delta.square().mean() + 0.001 * field['basis'][..., 2:].square().mean()
    total = mae + 0.25 * observation_nll + 0.3 * cross_rssi + 0.5 * rank + 0.6 * location + 0.2 * position_nll + 0.1 * tail_error + regularization
    if net.initial_posterior != 'none':
        total = total + device_region_weight * device_region_bias
    if not defer_cs_supervision:
        total = total + 0.15 * cs_nll + 0.1 * heldout_cs
    metrics = dict(loss=total.detach(), RSSI_MAE_db=mae.detach(), cross_RSSI_NLL=cross_rssi.detach(), location_median_m=position_error[query_valid].median().detach(), location_tail_m=tail_error.detach(), CS_NLL=cs_nll.detach(), heldout_CS_NLL=heldout_cs.detach(), ranking_loss=rank.detach(), radio_sigma_mean=sigma.mean().detach(), radio_outlier_mean=pi.mean().detach(), CS_sigma_mean=cs_sigma.mean().detach(), CS_outlier_mean=cs_pi.mean().detach(), short_residual_abs_db=delta.abs().mean().detach())
    if net.initial_posterior != 'none':
        metrics.update(device_only_RSSI_MAE_db=device_only_mae.detach(), heldout_device_region_bias_db=device_region_bias.detach())
    if joint_modes:
        metrics.update(location_median_m=mixture_median.detach(), mode_entropy=mixture_entropy.detach(), observation_NLL=observation_nll.detach(), radio_nominal_responsibility=nominal_mass.detach())
    if 'event_features' in batch:
        metrics.update(CS_event_labels=event_labels.detach(), causal_CS_feedback_events=event_feedback.detach())
    return (total, metrics)

def main():
    global _DATA
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--variant', choices=('set', 'pair'), required=True)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--epochs', type=int, default=16)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--functional-only', action='store_true')
    p.add_argument('--path-mixture', action='store_true')
    p.add_argument('--resume', type=Path)
    p.add_argument('--capture', type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--head-learning-rate', type=float)
    p.add_argument('--continuous-position', action='store_true')
    p.add_argument('--joint-modes', action='store_true')
    p.add_argument('--position-rule', choices=('mixed', 'hermite', 'balanced'), default='mixed')
    p.add_argument('--history-context', action='store_true')
    p.add_argument('--precision-head', action='store_true')
    p.add_argument('--robust-radio', action='store_true')
    p.add_argument('--robust-transfer', action='store_true')
    p.add_argument('--device-region-weight', type=float, default=0.3)
    p.add_argument('--peer-state-file', type=Path)
    p.add_argument('--initial-context-scans', type=int, choices=(1, 5), default=1)
    p.add_argument('--aligned-motion', action='store_true')
    p.add_argument('--defer-cs-supervision', action='store_true')
    p.add_argument('--full-pose-covariance', action='store_true')
    p.add_argument('--project-innovations', action='store_true')
    p.add_argument('--orientation-basis', action='store_true')
    p.add_argument('--orientation-learning-rate', type=float, default=0.003)
    p.add_argument('--mode-evidence', choices=('local', 'rolling', 'frozen'), default='local')
    p.add_argument('--initial-posterior', choices=('none', 'mean', 'full'), default='none')
    p.add_argument('--initial-learning-rate', type=float, default=0.001)
    p.add_argument('--ap-basis-prior', action='store_true')
    p.add_argument('--ap-basis-learning-rate', type=float, default=0.003)
    p.add_argument('--fixed-ap-basis', action='store_true')
    args = p.parse_args()
    if args.initial_posterior != 'none' and (not args.full_pose_covariance or args.initial_context_scans != 1):
        p.error('Initial recognition currently replaces exactly the first PRE full-covariance factor')
    torch.set_num_threads(2)
    if AVAILABLE_CPUS:
        os.sched_setaffinity(0, AVAILABLE_CPUS)
    torch.manual_seed(7901 if args.variant == 'set' else 8001)
    base, root = (args.base, args.root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((base / 'runs/fusion_network_v46_batched_wifi_candidates/configs/runtime_v46.json').read_text())
    data = base / 'data/processed_rssi_causal_short_v1'
    with np.load(data / 'source_map.npz', allow_pickle=False) as saved:
        cache = {k: saved[k] for k in saved.files}
    _DATA = BootstrapEpisodes(base, data, Path(cfg['trajectories']['manifest']).parent, args.capture or base / 'runs/mode_probability_v77_training/source_capture/capture', args.reference or base / 'runs/mode_probability_v77_training/reference_capture', cfg['anchors'], cache)
    write(root / 'source_training_data.json', _DATA.summary)
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork'))
    jobs = enqueue(pool, 790000, args.batch, args.workers)
    net = ThreeModuleNetwork(cache, cfg['pcrt']['source_only_propagation_geometry'], args.variant, args.history_context, args.precision_head, args.robust_radio, args.initial_context_scans, args.aligned_motion, args.full_pose_covariance, args.project_innovations, args.orientation_basis, args.mode_evidence, args.initial_posterior, args.ap_basis_prior, args.fixed_ap_basis, args.robust_transfer).cuda()
    if args.defer_cs_supervision:
        for module in [net.calibration.cs_encoder, net.calibration.cs_quality]:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    actor = torch.load(base / 'runs/fusion_network_v46_batched_wifi_candidates/outputs/pcrt_best.pt', map_location='cpu', weights_only=False)
    state = actor.get('state_dict', actor.get('model_state_dict', {}))
    motion = {k[len('decoder.motion_encoder.'):]: v for k, v in state.items() if k.startswith('decoder.motion_encoder.')}
    if motion:
        net.calibration.motion_encoder.load_state_dict(motion, strict=True)
    del actor, state
    write(root / 'architecture.json', net.configuration())
    with torch.no_grad():
        grid_fields = net.field.source(net.field.xy)
        eps = 0.15
        gradient = []
        for axis in range(2):
            delta = torch.zeros_like(net.field.xy)
            delta[:, axis] = eps
            gradient.append((net.field.source(net.field.xy + delta)['mean'] - net.field.source(net.field.xy - delta)['mean']) / (2 * eps))
        grid_fields['gradient'] = torch.stack(gradient, -1)
    anchors = torch.tensor(list(cfg['anchors'].values()), device='cuda', dtype=torch.float32)
    head_parameters = list(net.localization.parameters())
    head_ids = {id(v) for v in head_parameters}
    orientation_parameters = [net.field.orientation_strength] if args.orientation_basis else []
    initial_parameters = list(net.calibration.initial_encoder.parameters()) if args.initial_posterior != 'none' else []
    ap_basis_parameters = [net.field.ap_basis_strength] if args.ap_basis_prior else []
    auxiliary_ids = head_ids | {id(v) for v in orientation_parameters + initial_parameters + ap_basis_parameters}
    body_parameters = [v for v in net.parameters() if id(v) not in auxiliary_ids]
    previous = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    saved_split_groups = previous is not None and len(previous['optimizer']['param_groups']) >= 2
    head_replaced = previous is not None and args.precision_head and (previous['architecture'].get('localization_kind') != 'precision')
    orientation_added = previous is not None and args.orientation_basis and ('field.orientation_strength' not in previous['state_dict'])
    initializer_added = previous is not None and bool(initial_parameters) and ('calibration.initial_encoder.ap.0.weight' not in previous['state_dict'])
    ap_basis_added = previous is not None and args.ap_basis_prior and ('field.ap_basis_strength' not in previous['state_dict'])
    groups = [{'params': body_parameters}, {'params': head_parameters}]
    if orientation_parameters:
        groups.append({'params': orientation_parameters})
    if initial_parameters:
        groups.append({'params': initial_parameters})
    if ap_basis_parameters:
        groups.append({'params': ap_basis_parameters})
    optimizer = torch.optim.AdamW(groups if saved_split_groups or head_replaced or orientation_parameters or initial_parameters or ap_basis_parameters else net.parameters(), lr=0.0002, weight_decay=0.0001)
    optimizer_changed = previous is not None and [len(group['params']) for group in previous['optimizer']['param_groups']] != [len(group['params']) for group in optimizer.param_groups]
    start_epoch = 0
    if args.resume:
        previous_state = previous['state_dict']
        if head_replaced or orientation_added or initializer_added or ap_basis_added or optimizer_changed:
            net.load_state_dict({key: value for key, value in previous_state.items() if not (head_replaced and key.startswith('localization.'))}, strict=False)
            old_arch = previous['architecture']
            old_net = ThreeModuleNetwork(cache, cfg['pcrt']['source_only_propagation_geometry'], old_arch['variant'], old_arch.get('localization_context', False), old_arch.get('localization_kind') == 'precision', old_arch.get('radio_likelihood') == 'gaussian_student_t_mixture', old_arch.get('initial_context_scans', 1), old_arch.get('motion_history_policy') == 'wifi_only_transient_cs_v2', old_arch.get('calibration_geometry') == 'pre_pose_full_AP_covariance', old_arch.get('innovation_memory') == 'conditional_nuisance', old_arch.get('orientation_basis') == 'relative_AP_harmonics', old_arch.get('mode_evidence', 'local'), old_arch.get('initial_posterior', 'none'), old_arch.get('ap_basis_prior', False), old_arch.get('fixed_ap_basis', False), old_arch.get('robust_transfer', False))
            old_ids = [item for group in previous['optimizer']['param_groups'] for item in group['params']]
            old_names = previous.get('optimizer_parameter_names')
            old_names = [name for group in old_names for name in group] if old_names else list(dict(old_net.named_parameters()))
            previous_by_name = dict(zip(old_names, old_ids))
            for name, parameter in net.named_parameters():
                if name in previous_by_name and (not (head_replaced and name.startswith('localization.'))):
                    old_state = previous['optimizer']['state'].get(previous_by_name[name])
                    if old_state is not None:
                        optimizer.state[parameter] = dict(old_state)
            del old_net
        else:
            key = 'localization.ap.0.weight'
            if previous_state[key].shape != net.state_dict()[key].shape:
                old = previous_state[key]
                expanded = torch.zeros_like(net.state_dict()[key], device='cpu')
                if old.shape != (64, 16) or expanded.shape != (64, 72):
                    raise ValueError('Unsupported spatial head transition')
                expanded[:, :16] = old
                previous_state[key] = expanded
            net.load_state_dict(previous_state, strict=True)
            optimizer.load_state_dict(previous['optimizer'])
        start_epoch = previous['epoch'] + 1
        for parameter, state in optimizer.state.items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    if value.ndim and value.shape != parameter.shape:
                        if value.shape != (64, 16) or parameter.shape != (64, 72):
                            raise ValueError('Unsupported optimizer state transition')
                        expanded = torch.zeros_like(parameter, device='cpu')
                        expanded[:, :16] = value
                        value = expanded
                    state[key] = value.to('cuda')
        write(root / 'resume.json', dict(checkpoint=str(args.resume), start_epoch=start_epoch, optimizer_preserved=True, head_replaced=head_replaced, orientation_added=orientation_added, initializer_added=initializer_added, ap_basis_added=ap_basis_added, path_mixture=args.path_mixture))
    if args.learning_rate is not None:
        for group in optimizer.param_groups:
            group['lr'] = args.learning_rate
    if args.head_learning_rate is not None:
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['params'] = body_parameters
            optimizer.add_param_group(dict(params=head_parameters, lr=args.head_learning_rate))
        else:
            optimizer.param_groups[1]['lr'] = args.head_learning_rate
    if orientation_parameters:
        optimizer.param_groups[2]['lr'] = args.orientation_learning_rate
    if initial_parameters:
        optimizer.param_groups[2 + bool(orientation_parameters)]['lr'] = args.initial_learning_rate
    if ap_basis_parameters:
        optimizer.param_groups[2 + bool(orientation_parameters) + bool(initial_parameters)]['lr'] = args.ap_basis_learning_rate
    if args.path_mixture:
        old_losses = globals()['losses']
        globals()['losses'] = lambda *values: old_losses(*values, path_mixture=True, continuous_position=args.continuous_position, joint_modes=args.joint_modes, position_rule=args.position_rule, defer_cs_supervision=args.defer_cs_supervision, device_region_weight=args.device_region_weight)
    small = _DATA.sample(np.random.default_rng(791), batch=2, device='cuda')
    loss, metrics = losses(net, small, grid_fields, anchors)
    loss.backward()
    groups = {name: sum((float(v.grad.detach().float().square().sum()) for v in module.parameters() if v.grad is not None)) ** 0.5 for name, module in [('M1', net.field), ('M2', net.calibration), ('M3', net.localization)]}
    if _DATA.event_source.phones and (not args.defer_cs_supervision):
        groups['CS_event_quality'] = sum((float(v.grad.detach().float().square().sum()) for module in [net.calibration.cs_encoder, net.calibration.cs_quality] for v in module.parameters() if v.grad is not None)) ** 0.5
    if args.orientation_basis:
        groups['M1_heading_harmonics'] = float(net.field.orientation_strength.grad.detach().norm())
    if initial_parameters:
        groups['M2_initial_encoder'] = sum((float(value.grad.detach().float().square().sum()) for value in initial_parameters if value.grad is not None)) ** 0.5
    if ap_basis_parameters and (not args.fixed_ap_basis):
        groups['M1_AP_basis_prior'] = float(net.field.ap_basis_strength.grad.detach().abs())
    prefix_difference = None
    if initial_parameters:
        prefix_difference = dict(mean=float((net.calibration.last_initial_prefix_mean - small['prefix_device_mean']).abs().max()), covariance=float((net.calibration.last_initial_prefix_covariance - small['prefix_device_covariance']).abs().max()))
        if prefix_difference['mean'] > 0.02 or prefix_difference['covariance'] > 0.05:
            raise RuntimeError('Initial recognition source/runtime prefix mismatch: ' + str(prefix_difference))
    if not torch.isfinite(loss) or not all((np.isfinite(v) and v > 0 for v in groups.values())):
        raise RuntimeError('Actual source functional path failed: ' + str(groups))
    optimizer.zero_grad(set_to_none=True)
    write(root / 'functional_path.json', dict(status='PASS_SOURCE_FORWARD_BACKWARD', gradient_norms=groups, test_devices_opened=False, labels_only_in_loss=True, initial_prefix_difference=prefix_difference, loss=float(loss.detach())))
    print(json.dumps(dict(functional='PASS', gradients=groups)), flush=True)
    if args.functional_only:
        pool.shutdown(wait=True)
        return
    history = []
    seed = 791000
    started = time.monotonic()
    effective_batch = args.batch
    for epoch in range(start_epoch, args.epochs + (32 if args.peer_state_file else 0)):
        if epoch >= args.epochs and (not peer_requires_training(args.peer_state_file)):
            break
        began = time.monotonic()
        rows = []
        for step in range(args.steps):
            chunks = [job.result() for job in jobs]
            tensors = {k: torch.from_numpy(np.concatenate([c[k] for c in chunks], axis=0)).cuda(non_blocking=True) for k in chunks[0]}
            jobs = enqueue(pool, seed, effective_batch, args.workers)
            seed += args.workers
            torch.cuda.reset_peak_memory_stats()
            while True:
                try:
                    row = perform_update(net, tensors, grid_fields, anchors, optimizer)
                    break
                except torch.cuda.OutOfMemoryError:
                    count = len(tensors['rssi']) // 2
                    if count < 12:
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    tensors = {k: v[:count * 4] if k.startswith('event_') else v[:count] for k, v in tensors.items()}
                    effective_batch = count
                    print(json.dumps(dict(event='REDUCE_ACTUAL_TRAINING_BATCH_AFTER_OOM', batch=count)), flush=True)
            free, total = torch.cuda.mem_get_info()
            external = max(0, total - free - torch.cuda.memory_reserved())
            peak = torch.cuda.max_memory_allocated()
            suggested = int(len(tensors['rssi']) * (0.79 * total - external) / max(peak, 1))
            quantum = 4 if args.joint_modes else 12
            suggested = max(12, min(768, math.ceil(suggested / quantum) * quantum))
            if peak + external < 0.75 * total or peak + external > 0.86 * total:
                effective_batch = min(suggested, int(len(tensors['rssi']) * 1.5))
            row.update(next_batch=effective_batch, GPU_observed_used_fraction=(total - free) / total)
            rows.append(row)
            write(root / 'live_training.json', dict(status='RUNNING', phase='SOURCE_NATIVE_PREFIX_JOINT' if args.capture else 'SOURCE_BOOTSTRAP', epoch=epoch, step=step, variant=args.variant, pid=os.getpid(), native_prefix_training=bool(args.capture), continuous_position=args.continuous_position, joint_mode_training=args.joint_modes, position_rule=args.position_rule, **row))
            print(json.dumps(dict(epoch=epoch, step=step, **row)), flush=True)
            del tensors
        summary = {k: float(np.mean([v[k] for v in rows])) for k in rows[0]}
        summary.update(epoch=epoch, seconds=time.monotonic() - began)
        history.append(summary)
        write(root / 'training_history.json', history)
        checkpoint = dict(state_dict=net.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, architecture=net.configuration(), phase='SOURCE_PATH_MIXTURE_JOINT' if args.path_mixture else 'SOURCE_BOOTSTRAP_NOT_NATIVE_ACCEPTANCE', data_root=str(data), source_only=True, continuous_position_training=args.continuous_position, joint_mode_training=args.joint_modes, position_rule_training=args.position_rule, training_contract=dict(joint_normalized_radio_density=bool(args.precision_head and args.joint_modes), causal_native_short_residual_in_support=bool(args.capture), counterfactual_short_residual_frozen_at_region_entry=bool(args.capture), CS_supervision_deferred=args.defer_cs_supervision, CS_event_aligned_supervision=bool(_DATA.event_source.phones and (not args.defer_cs_supervision)), full_AP_pose_covariance=args.full_pose_covariance, projected_nuisance_memory=args.project_innovations, causal_AP_bearing_harmonics=args.orientation_basis, regional_mode_evidence=args.mode_evidence, initial_recognition_posterior=args.initial_posterior, source_heldout_device_DC_objective=args.initial_posterior != 'none' and args.device_region_weight > 0, source_heldout_device_DC_weight=args.device_region_weight, explicit_AP_bias_basis=args.ap_basis_prior, region_prior_before_first_observation=True, regional_Bayes_factor_untempered=True))
        parameter_names = {id(value): name for name, value in net.named_parameters()}
        checkpoint['optimizer_parameter_names'] = [[parameter_names[id(value)] for value in group['params']] for group in optimizer.param_groups]
        temporary = root / 'last.pending.pt'
        torch.save(checkpoint, temporary)
        temporary.replace(root / 'last.pt')
        if epoch % 4 == 3 or epoch == args.epochs - 1:
            torch.save(checkpoint, root / ('epoch_%03d.pt' % epoch))
        if epoch == args.epochs - 1:
            write(root / 'nominal_training_complete.json', dict(pid=os.getpid(), epoch=epoch, checkpoint=str(root / ('epoch_%03d.pt' % epoch)), source_only=True))
    pool.shutdown(wait=True)
    final = json.loads((root / 'live_training.json').read_text())
    final['status'] = 'COMPLETE'
    write(root / 'live_training.json', final)
    write(root / 'bootstrap_complete.json', dict(status='COMPLETE', phase='SOURCE_NATIVE_PREFIX_CONDITIONAL' if args.capture else 'SOURCE_BOOTSTRAP', epochs=args.epochs, actual_final_epoch=final['epoch'], seconds=time.monotonic() - started, native_training_required=True, goal_complete=False, variant=args.variant))
if __name__ == '__main__':
    main()
