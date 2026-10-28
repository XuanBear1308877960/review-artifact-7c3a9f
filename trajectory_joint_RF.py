import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.special import gammaln
from engine_utils import write

def unpack_auxiliary(parameters):
    p = np.asarray(parameters, float)
    return dict(gain=p[:2], anchor_mean=p[2:10].reshape(4, 2), alias=np.vstack((np.zeros(2), p[10:14].reshape(2, 2))), beta=p[14:22].reshape(4, 2), sigma=np.exp(p[22:30].reshape(4, 2)), rho=0.95 * np.tanh(p[30:34]), background=p[34:42].reshape(4, 2))

def conditional_numpy(residual, valid, sigma, rho, derivative=False):
    m = np.asarray(valid, bool)
    r = residual * m
    both = m.all(-1)
    k = m.sum(-1)
    s = np.broadcast_to(sigma, r.shape)
    rho = np.broadcast_to(rho, r.shape[:-1])
    den = 1 - rho * rho
    w = r / (s * s)
    w_both = np.stack(((r[..., 0] / s[..., 0] ** 2 - rho * r[..., 1] / (s[..., 0] * s[..., 1])) / den, (r[..., 1] / s[..., 1] ** 2 - rho * r[..., 0] / (s[..., 0] * s[..., 1])) / den), -1)
    w = np.where(both[..., None], w_both, w) * m
    v = (r * w).sum(-1)
    ld = (2 * np.log(s) * m).sum(-1) + np.where(both, np.log(den), 0.0)
    nll = gammaln(2) - gammaln((4 + k) / 2) + k / 2 * math.log(4 * math.pi) + 0.5 * ld + (4 + k) / 2 * np.log1p(v / 4)
    if not derivative:
        return nll
    return (nll, (4 + k)[..., None] / (4 + v)[..., None] * w)

def auxiliary_emission(distance, error, context, derivative=False):
    p = unpack_auxiliary(context['auxiliary_parameters'])
    anchor = context['anchor']
    expected = np.asarray(distance)[:, None, None] * p['gain'][None, None, :] + p['anchor_mean'][anchor][None, None, :] + p['alias'][None]
    expected = np.concatenate((expected, np.broadcast_to(p['background'][anchor], (len(distance), 1, 2))), 1)
    expected += np.asarray(context.get('auxiliary_bias_m', [0.0, 0.0]))[None, None]
    residual = context['auxiliary_values'][:, None, :] - expected - p['beta'][None] * error[:, :, None]
    valid = np.broadcast_to(context['auxiliary_valid'][:, None, :], residual.shape)
    out = conditional_numpy(residual, valid, p['sigma'][None], p['rho'][None], derivative)
    if not derivative:
        return -out
    loss, score = out
    g0 = context['parameters']['gain']
    context['last_auxiliary_score'] = score
    slope = np.concatenate((-p['gain'][None] + p['beta'][:3] * g0, np.zeros((1, 2))), 0)
    return (-loss, (score * slope[None]).sum(-1))

def null_auxiliary_energy(context):
    z = np.asarray(context['auxiliary_null_parameters'], float)
    anchor = context['anchor']
    mean = z[2:10].reshape(4, 2)[anchor] + z[:2] * (context['raw'] - context['background'])[:, None]
    return float(conditional_numpy(context['auxiliary_values'] - mean, context['auxiliary_valid'], np.exp(z[10:12]), 0.95 * math.tanh(z[12])).sum())

def arrays(a, prior):
    from trajectory_dual_cs import source_blocks
    from trajectory_raw_hmm import unpack
    rows = source_blocks(a, prior)
    p = unpack(prior['raw_HMM_parameters'])
    B = len(rows)
    M = max((len(r['raw']) for r in rows))
    raw = np.zeros((B, M))
    truth = raw.copy()
    mad = raw.copy()
    dt = np.ones_like(raw)
    mask = np.zeros_like(raw, bool)
    auxiliary = np.zeros((B, M, 2))
    valid = np.zeros_like(auxiliary, bool)
    anchor = []
    held = []
    for i, b in enumerate(rows):
        n = len(b['raw'])
        raw[i, :n] = b['raw']
        truth[i, :n] = b['true_raw']
        mad[i, :n] = b['raw_scale']
        mask[i, :n] = True
        if n > 1:
            dt[i, 1:n] = np.diag(b['dt'], k=-1)
        with np.load(a.queries / 'source_train' / b['trajectory_id'] / 'CS_history_queries.npz', allow_pickle=False) as z:
            ix = b['event_indices']
            auxiliary[i, :n] = z['auxiliary_values'][ix, :2]
            valid[i, :n] = z['auxiliary_is_finite'][ix, :2] & (auxiliary[i, :n] > 0)
            valid[i, :n, 1] &= z['auxiliary_values'][ix, 3] > 0
        anchor.append(b['anchor'])
        held.append(b['heldout'])
    anchor = np.asarray(anchor)
    held = np.asarray(held, bool)
    error = raw[:, :, None] - p['gain'] * truth[:, :, None] - p['anchor_mean'][anchor, None, None] - p['alias'][None, None]
    error = np.concatenate((error, (raw - np.asarray(prior['raw_HMM_background_range_m'])[anchor, None])[:, :, None]), -1)
    sigma = np.sqrt(p['sigma'][None, None, :3] ** 2 + (p['mad_coefficient'] * mad[:, :, None]) ** 2)
    sigma = np.concatenate((sigma, np.full((B, M, 1), p['sigma'][3])), -1)
    return dict(raw=raw, raw_scale=mad, truth=truth, error=error, sigma=sigma, dt=dt, mask=mask, auxiliary=auxiliary, valid=valid, anchor=anchor, held=held)

def fit(a):
    import torch
    from base_evaluate import AVAILABLE_CPUS
    from trajectory_raw_hmm import unpack
    import os
    torch.set_num_threads(1)
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    p = unpack(prior['raw_HMM_parameters'])
    d = arrays(a, prior)
    x = {k: torch.tensor(v, dtype=torch.bool if v.dtype == bool else torch.long if k == 'anchor' else torch.float64) for k, v in d.items()}
    train = ~x['held']
    held = x['held']
    e = x['error']
    sigma = x['sigma']
    mask = x['mask']
    valid = x['valid']
    y = x['auxiliary']
    anchor = x['anchor']
    truth = x['truth']
    emission0 = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi) - sigma.log() - 2 * torch.log1p(e.square() / (3 * sigma.square()))
    pi = torch.tensor(p['probability'], dtype=torch.float64)
    tau = torch.tensor(p['persistence_s'], dtype=torch.float64)

    def auxiliary_loss(residual, valid, sig, rho):
        m = valid.to(torch.float64)
        r = residual * m
        both = valid.all(-1)
        k = m.sum(-1)
        den = 1 - rho.square()
        w = r / sig.square()
        wb = torch.stack(((r[..., 0] / sig[..., 0].square() - rho * r[..., 1] / (sig[..., 0] * sig[..., 1])) / den, (r[..., 1] / sig[..., 1].square() - rho * r[..., 0] / (sig[..., 0] * sig[..., 1])) / den), -1)
        w = torch.where(both[..., None], wb, w) * m
        v = (r * w).sum(-1)
        ld = (2 * sig.log() * m).sum(-1) + torch.where(both, den.log(), torch.zeros_like(v))
        return math.lgamma(2) - torch.lgamma((4 + k) / 2) + k / 2 * math.log(4 * math.pi) + 0.5 * ld + (4 + k) / 2 * torch.log1p(v / 4)

    def sequence(emission):
        alpha = pi.log()[None] + emission[:, 0]
        for j in range(1, e.shape[1]):
            stick = (-x['dt'][:, j, None] / tau[None]).exp() if prior['raw_HMM_temporal_selected'] else torch.zeros((len(e), 4), dtype=torch.float64)
            transition = (1 - stick[:, :, None]) * pi[None, None] + torch.diag_embed(stick)
            proposed = torch.logsumexp(alpha[:, :, None] + transition.clamp_min(1e-15).log(), 1) + emission[:, j]
            alpha = torch.where(mask[:, j, None], proposed, alpha)
        return -torch.logsumexp(alpha, -1)
    raw_nll = sequence(emission0).detach()
    medians = np.asarray([[np.median(d['auxiliary'][(d['anchor'] == j) & ~d['held'], :, c][d['valid'][(d['anchor'] == j) & ~d['held'], :, c]]) for c in range(2)] for j in range(4)])
    null_initial = np.r_[np.zeros(2), medians.ravel(), np.log([5.0, 5.0]), 0.0]
    null = torch.nn.Parameter(torch.tensor(null_initial, dtype=torch.float64))
    opt = torch.optim.Adam([null], lr=0.035)
    best_null = None

    def null_losses(parameter):
        mean = parameter[2:10].reshape(4, 2)[anchor, None] + parameter[:2][None, None] * (x['raw'] - torch.tensor(prior['raw_HMM_background_range_m'], dtype=torch.float64)[anchor, None])[:, :, None]
        loss = auxiliary_loss(y - mean, valid, parameter[10:12].exp(), 0.95 * parameter[12].tanh())
        return raw_nll + (loss * mask).sum(-1)
    for update in range(401):
        losses = null_losses(null)
        if update % 20 == 0:
            val = float(losses[held].sum() / mask[held].sum())
            if best_null is None or val < best_null[0]:
                best_null = (val, null.detach().numpy().copy(), update)
        if update == 400:
            break
        opt.zero_grad()
        (losses[train].sum() / mask[train].sum()).backward()
        opt.step()
    z = best_null[1]
    beta = z[:2]
    bg = z[2:10].reshape(4, 2)
    initial = np.r_[beta * p['gain'], (bg + beta[None] * (p['anchor_mean'] - np.asarray(prior['raw_HMM_background_range_m']))[:, None]).ravel(), (p['alias'][1:, None] * beta[None]).ravel(), np.tile(beta, 4), np.tile(z[10:12], 4), np.full(4, z[12]), bg.ravel()]
    lower = np.r_[[-4.0] * 2, [-60.0] * 8, [-30.0] * 4, [-4.0] * 8, [math.log(0.2)] * 8, [-3.0] * 4, [-60.0] * 8]
    upper = np.r_[[4.0] * 2, [80.0] * 8, [30.0] * 4, [4.0] * 8, [math.log(50.0)] * 8, [3.0] * 4, [80.0] * 8]
    parameter = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=0.025)
    best = None
    trace = []
    observed = valid[:, :, None, :].expand(-1, -1, 4, -1)

    def full_losses(parameter):
        means = truth[:, :, None, None] * parameter[:2][None, None, None] + parameter[2:10].reshape(4, 2)[anchor, None, None] + torch.cat((torch.zeros((1, 2), dtype=torch.float64), parameter[10:14].reshape(2, 2)))[None, None]
        means = torch.cat((means, parameter[34:42].reshape(4, 2)[anchor, None, None].expand(-1, truth.shape[1], 1, -1)), 2)
        residual = y[:, :, None] - means - parameter[14:22].reshape(4, 2)[None, None] * e[:, :, :, None]
        loss = auxiliary_loss(residual, observed, parameter[22:30].reshape(4, 2).exp()[None, None], 0.95 * parameter[30:34].tanh()[None, None])
        return sequence(emission0 - loss)
    nested = float(full_losses(parameter)[held].sum() / mask[held].sum())
    if abs(nested - best_null[0]) > 1e-07:
        raise ValueError('Position-independent auxiliary density is not nested exactly')
    if AVAILABLE_CPUS:
        os.sched_setaffinity(0, AVAILABLE_CPUS)
    for update in range(801):
        losses = full_losses(parameter)
        if update % 20 == 0:
            val = float(losses[held].sum() / mask[held].sum())
            row = dict(update=update, source_train_NLL=float(losses[train].sum() / mask[train].sum()), source_heldout_NLL=val)
            trace.append(row)
            if best is None or val < best[0]:
                best = (val, parameter.detach().numpy().copy(), update)
        if update == 800:
            break
        objective = losses[train].sum() / mask[train].sum() + 5e-05 * (parameter - torch.tensor(initial)).square().sum()
        optimizer.zero_grad()
        objective.backward()
        torch.nn.utils.clip_grad_norm_([parameter], 10.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(torch.tensor(lower), torch.tensor(upper))
    selected = best[0] < best_null[0] - 1e-06
    fitted = unpack_auxiliary(best[1])
    with torch.no_grad():
        family_full = full_losses(torch.tensor(best[1], dtype=torch.float64)).detach()
        family_null = null_losses(torch.tensor(best_null[1], dtype=torch.float64)).detach()
    logits = torch.nn.Parameter(torch.full((4,), math.log(9.0), dtype=torch.float64))
    family_optimizer = torch.optim.Adam([logits], lr=0.06)

    def family_losses():
        return -torch.logaddexp(torch.nn.functional.logsigmoid(logits)[anchor] - family_full, torch.nn.functional.logsigmoid(-logits)[anchor] - family_null)
    for update in range(301):
        losses = family_losses()
        if update == 300:
            break
        hyperprior = -(torch.nn.functional.logsigmoid(logits) + torch.nn.functional.logsigmoid(-logits)).sum()
        objective = (losses[train].sum() + hyperprior) / mask[train].sum()
        family_optimizer.zero_grad()
        objective.backward()
        family_optimizer.step()
    family_validation = float(family_losses()[held].sum() / mask[held].sum())
    family_selected = family_validation < best[0]
    from trajectory_raw_hmm import block_context, raw_hmm_energy
    i = 0
    n = int(d['mask'][i].sum())
    t = np.r_[0, np.cumsum(d['dt'][i, 1:n])]
    j = int(d['anchor'][i])
    context = block_context(j, t, d['raw'][i, :n], d['raw_scale'][i, :n], prior['raw_HMM_parameters'], prior['raw_HMM_background_range_m'], prior['raw_HMM_temporal_selected'])
    context.update(auxiliary_parameters=best[1], auxiliary_values=d['auxiliary'][i, :n], auxiliary_valid=d['valid'][i, :n])
    direction = np.random.RandomState(82553).normal(size=n)
    direction /= np.linalg.norm(direction)
    h = 1e-05
    distance = d['truth'][i, :n]
    _, gradient, _ = raw_hmm_energy(distance, context, True)
    numeric = (raw_hmm_energy(distance + h * direction, context) - raw_hmm_energy(distance - h * direction, context)) / (2 * h)
    gradient_error = float(abs(numeric - gradient @ direction))
    if gradient_error > 3e-05:
        raise ValueError('Joint RF source derivative mismatch')
    family_gradient_error = None
    if family_selected:
        context.update(auxiliary_null_parameters=best_null[1], auxiliary_family_probability=float(torch.sigmoid(logits)[j]))
        context['auxiliary_null_loss'] = null_auxiliary_energy(context)
        _, gradient, _ = raw_hmm_energy(distance, context, True)
        numeric = (raw_hmm_energy(distance + h * direction, context) - raw_hmm_energy(distance - h * direction, context)) / (2 * h)
        family_gradient_error = float(abs(numeric - gradient @ direction))
        if family_gradient_error > 3e-05:
            raise ValueError('Joint RF family source derivative mismatch')
    prior.update(joint_RF_auxiliary_selected=bool(selected), joint_RF_auxiliary_parameters=best[1].tolist(), joint_RF_auxiliary_query_root=str(a.queries.resolve()), joint_RF_auxiliary_source_heldout_NLL=best[0], joint_RF_position_independent_auxiliary_source_heldout_NLL=best_null[0], joint_RF_source_selected_update=best[2], joint_RF_source_trace=trace, joint_RF_target_GT_used=False, joint_RF_source_training_43_selection_11=True, joint_RF_raw_noise_scales_frozen=True, joint_RF_auxiliary_fit_seconds=time.monotonic() - began)
    prior.update(joint_RF_null_auxiliary_parameters=best_null[1].tolist(), joint_RF_family_mixture_selected=bool(family_selected), joint_RF_family_probability=torch.sigmoid(logits).detach().numpy().tolist(), joint_RF_family_source_gradient_difference=family_gradient_error, joint_RF_family_source_heldout_NLL=family_validation, joint_RF_family_scope='One family per anchor trajectory; common measured channels, no duplicate independent factor')
    prior['joint_RF_source_gradient_difference'] = gradient_error
    prior.pop('raw_HMM_context_root', None)
    write(a.output, prior)
    print(json.dumps(dict(selected=selected, heldout_NLL=best[0], position_independent_auxiliary_NLL=best_null[0], selected_update=best[2], auxiliary_gain=fitted['gain'].tolist(), conditional_noise_sigma_m=fitted['sigma'].tolist(), conditional_correlation=fitted['rho'].tolist(), family_mixture_selected=bool(family_selected), family_heldout_NLL=family_validation, family_probability=torch.sigmoid(logits).detach().numpy().tolist(), seconds=time.monotonic() - began)), flush=True)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'prior', 'queries', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    fit(p.parse_args())
if __name__ == '__main__':
    main()
