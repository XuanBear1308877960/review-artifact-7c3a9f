import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp, softmax
from engine_utils import write
from trajectory_dual_cs import DualAxialPath, source_blocks
from trajectory_force_marginal import ForceDualPath
from trajectory_path import SensorPath

def unpack(parameters):
    p = np.asarray(parameters, float)
    return dict(gain=math.exp(p[0]), anchor_mean=p[1:5], alias=np.array([0.0, math.exp(p[5]), -math.exp(p[6])]), sigma=np.exp(p[7:11]), mad_coefficient=math.exp(p[11]), persistence_s=np.exp(p[12:16]), probability=softmax(np.r_[p[16:19], 0.0]))

def block_context(anchor, t, raw, scale, parameters, background, temporal=True, probabilities=None, conditional_bias_m=None):
    p = unpack(parameters)
    t = np.asarray(t, float)
    sigma = np.sqrt(p['sigma'][None, :3] ** 2 + (p['mad_coefficient'] * np.asarray(scale, float)[:, None]) ** 2)
    sigma = np.c_[sigma, np.full(len(t), p['sigma'][3])]
    log_transition = []
    probabilities = np.broadcast_to(p['probability'], (len(t), 4)) if probabilities is None else np.asarray(probabilities, float)
    for i, dt in enumerate(np.diff(t)):
        stick = np.exp(-max(float(dt), 1e-05) / p['persistence_s']) if temporal else np.zeros(4)
        transition = (1 - stick[:, None]) * probabilities[i + 1, None, :] + np.diag(stick)
        log_transition.append(np.log(transition.clip(1e-15)))
    offset = np.zeros(len(t)) if conditional_bias_m is None else np.asarray(conditional_bias_m, float)
    if offset.shape != (len(t),) or not np.isfinite(offset).all():
        raise ValueError('Invalid conditional range offset')
    return dict(anchor=anchor, raw=np.asarray(raw, float), sigma=sigma, conditional_bias_m=offset, log_transition=np.asarray(log_transition), transition=np.exp(np.asarray(log_transition)), log_initial=np.log(probabilities[0].clip(1e-15)), background=float(background[anchor]), parameters=p)

def raw_hmm_energy(distance, context, derivative=False, bias_m=0.0):
    if 'auxiliary_family_probability' in context:
        probability = float(context['auxiliary_family_probability'])
        full_context = dict(context)
        full_context.pop('auxiliary_family_probability')
        null_context = dict(full_context)
        null_context.pop('auxiliary_parameters')
        full = raw_hmm_energy(distance, full_context, derivative, bias_m)
        null = raw_hmm_energy(distance, null_context, derivative, bias_m)
        losses = np.asarray([full[0] if derivative else full, (null[0] if derivative else null) + context['auxiliary_null_loss']])
        logmass = np.log([probability, 1 - probability]) - losses
        normalizer = logsumexp(logmass)
        weight = np.exp(logmass - normalizer)
        context['last_auxiliary_family_posterior'] = float(weight[0])
        if not derivative:
            return -float(normalizer)
        return (-float(normalizer), weight[0] * full[1] + weight[1] * null[1], weight[0] * full[2] + weight[1] * null[2])
    p = context['parameters']
    raw = context['raw']
    sigma = context['sigma']
    e = raw[:, None] - p['gain'] * np.asarray(distance)[:, None] - p['anchor_mean'][context['anchor']] - p['alias'][None, :] - bias_m
    if 'conditional_bias_m' in context:
        e -= context['conditional_bias_m'][:, None]
    e = np.c_[e, raw - context['background']]
    emission = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi) - np.log(sigma) - 2 * np.log1p(e * e / (3 * sigma * sigma))
    auxiliary_gradient = None
    if 'auxiliary_parameters' in context:
        from trajectory_joint_RF import auxiliary_emission
        if derivative:
            extra, auxiliary_gradient = auxiliary_emission(distance, e, context, True)
        else:
            extra = auxiliary_emission(distance, e, context)
        emission += extra
    maximum = emission.max(-1)
    density = np.exp(emission - maximum[:, None])
    alpha = np.empty_like(emission)
    normalizer = np.empty(len(e))
    alpha[0] = np.exp(context['log_initial']) * density[0]
    normalizer[0] = alpha[0].sum()
    alpha[0] /= normalizer[0]
    for i in range(1, len(e)):
        alpha[i] = alpha[i - 1] @ context['transition'][i - 1] * density[i]
        normalizer[i] = alpha[i].sum()
        alpha[i] /= normalizer[i]
    loss = -float((np.log(normalizer) + maximum).sum())
    if not derivative:
        return loss
    beta = np.ones_like(emission)
    for i in range(len(e) - 2, -1, -1):
        beta[i] = context['transition'][i] @ (density[i + 1] * beta[i + 1]) / normalizer[i + 1]
    posterior = alpha * beta
    posterior /= posterior.sum(-1, keepdims=True)
    score = 4 * e / (3 * sigma * sigma + e * e)
    gradient = -p['gain'] * (posterior[:, :3] * score[:, :3]).sum(-1)
    if auxiliary_gradient is not None:
        gradient += (posterior * auxiliary_gradient).sum(-1)
        context['last_auxiliary_bias_gradient'] = -(posterior[:, :, None] * context['last_auxiliary_score']).sum((0, 1))
    return (loss, gradient, posterior)

def fit(a):
    import torch
    torch.set_num_threads(1)
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    rows = source_blocks(a, prior)
    background = [float(np.median(np.concatenate([b['raw'] for b in rows if b['anchor'] == j and (not b['heldout'])]))) for j in range(4)]
    max_n = max((len(b['raw']) for b in rows))
    B = len(rows)
    raw = np.zeros((B, max_n))
    truth = raw.copy()
    mad = raw.copy()
    dt = np.ones_like(raw)
    valid = raw.copy()
    anchor = []
    heldout = []
    for i, b in enumerate(rows):
        n = len(b['raw'])
        raw[i, :n] = b['raw']
        truth[i, :n] = b['true_raw']
        mad[i, :n] = b['raw_scale']
        valid[i, :n] = 1
        if n > 1:
            dt[i, 1:n] = np.diag(b['dt'], k=-1)
        anchor.append(b['anchor'])
        heldout.append(b['heldout'])
    raw, truth, mad, dt, valid = [torch.tensor(x, dtype=torch.float64) for x in (raw, truth, mad, dt, valid)]
    anchor = torch.tensor(anchor, dtype=torch.long)
    heldout = torch.tensor(heldout, dtype=torch.bool)
    bg = torch.tensor(background, dtype=torch.float64)[anchor]
    initial = np.r_[prior['dual_CS_parameters'][6], np.asarray(prior['dual_CS_parameters'])[8:12], np.log([2.5, 4.0]), np.log([0.45, 1.2, 2.5, 8.0]), math.log(0.25), np.log([2.0, 2.0, 2.0, 0.7]), np.log(np.array([0.6, 0.16, 0.12]) / 0.12)]
    bounds = [(math.log(0.7), math.log(1.4))] + [(-5.0, 5.0)] * 4 + [(math.log(0.3), math.log(15.0))] * 2
    bounds += [(math.log(0.1), math.log(10.0))] * 3 + [(math.log(4.0), math.log(25.0)), (math.log(0.01), math.log(3.0))]
    bounds += [(math.log(0.05), math.log(20.0))] * 4 + [(-6.0, 6.0)] * 3
    low = torch.tensor([x[0] for x in bounds])
    high = torch.tensor([x[1] for x in bounds])
    mask_train = ~heldout
    mask_val = heldout

    def likelihood(p, temporal):
        gain = p[0].exp()
        mean = p[1:5][anchor]
        alias = torch.stack((p[5] * 0, p[5].exp(), -p[6].exp()))
        error = raw[:, :, None] - gain * truth[:, :, None] - mean[:, None, None] - alias[None, None, :]
        error = torch.cat((error, (raw - bg[:, None])[:, :, None]), -1)
        sigma3 = torch.sqrt(p[7:10].exp()[None, None, :] ** 2 + (p[11].exp() * mad[:, :, None]) ** 2)
        sigma = torch.cat((sigma3, torch.ones_like(raw)[:, :, None] * p[10].exp()), -1)
        emission = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi) - torch.log(sigma) - 2 * torch.log1p(error.square() / (3 * sigma.square()))
        probability = torch.softmax(torch.cat((p[16:19], p[16:17] * 0)), 0)
        alpha = probability.log()[None, :] + emission[:, 0, :]
        for j in range(1, max_n):
            if temporal:
                stick = torch.exp(-dt[:, j, None] / p[12:16].exp()[None, :])
                transition = (1 - stick[:, :, None]) * probability[None, None, :] + torch.diag_embed(stick)
                proposed = torch.logsumexp(alpha[:, :, None] + transition.clamp_min(1e-15).log(), dim=1) + emission[:, j, :]
            else:
                proposed = torch.logsumexp(alpha, dim=-1)[:, None] + probability.log()[None, :] + emission[:, j, :]
            alpha = torch.where(valid[:, j, None].bool(), proposed, alpha)
        loss = -torch.logsumexp(alpha, dim=-1)
        return loss
    results = {}
    for temporal in (False, True):
        p = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float64))
        opt = torch.optim.Adam([p], lr=0.035)
        best = None
        trace = []
        for step in range(601):
            losses = likelihood(p, temporal)
            if step % 20 == 0:
                validation = float(losses[mask_val].sum() / valid[mask_val].sum())
                train = float(losses[mask_train].sum() / valid[mask_train].sum())
                row = dict(update=step, train_NLL=train, heldout_NLL=validation)
                trace.append(row)
                if best is None or validation < best[0]:
                    best = (validation, p.detach().numpy().copy(), step)
            if step == 600:
                break
            objective = losses[mask_train].sum() / valid[mask_train].sum()
            objective += 0.002 * (p[0] / 0.2).square() + 0.0002 * (p[1:5] - torch.tensor(initial[1:5])).square().sum()
            opt.zero_grad()
            objective.backward()
            opt.step()
            with torch.no_grad():
                p.clamp_(low, high)
        name = 'temporal' if temporal else 'independent'
        results[name] = dict(heldout_NLL=best[0], parameters=best[1].tolist(), selected_update=best[2], trace=trace)
        print(json.dumps(dict(model=name, heldout_NLL=best[0], selected_update=best[2], seconds=time.monotonic() - began)), flush=True)
    temporal = results['temporal']['heldout_NLL'] < results['independent']['heldout_NLL']
    chosen = results['temporal' if temporal else 'independent']
    p = unpack(chosen['parameters'])
    prior.update(raw_HMM_parameters=chosen['parameters'], raw_HMM_background_range_m=background, raw_HMM_temporal_selected=bool(temporal), raw_HMM_source_fits=results, raw_HMM_source_only=True, raw_HMM_later_GT_used=False, raw_HMM_scope='Final direct raw CS factor. Frozen M2/proposals still use original corrected ranges.', raw_HMM_persistence_s=p['persistence_s'].tolist(), raw_HMM_component_probability=p['probability'].tolist(), raw_HMM_fit_seconds=time.monotonic() - began)
    write(a.output, prior)
    print(json.dumps(dict(status='SOURCE_RAW_RANGE_MODEL_FIT_COMPLETE', temporal_selected=temporal, alias_offset_m=p['alias'].tolist(), sigma_m=p['sigma'].tolist(), persistence_s=p['persistence_s'].tolist(), component_probability=p['probability'].tolist(), source_heldout_NLL=chosen['heldout_NLL'], seconds=time.monotonic() - began)), flush=True)

class RawHMMPath(DualAxialPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hmm = []
        from trajectory_dual_cs import raw_scale
        quality = None
        conditional_bias = None
        auxiliary = None
        if 'raw_HMM_context_root' in self.prior:
            with np.load(Path(self.prior['raw_HMM_context_root']) / args[0].parent.name / args[0].name / 'raw_HMM_context.npz', allow_pickle=False) as z:
                if not np.array_equal(z['anchor_index'], self.anchor_index):
                    raise ValueError('HMM context anchor mismatch')
                if not np.allclose((z['timestamp_ns'] - self.s['start_ns']) / 1000000000.0, self.ct, rtol=0, atol=1e-08):
                    raise ValueError('HMM context time mismatch')
                quality = z['component_probability'].copy()
                if 'conditional_bias_m' in z.files:
                    conditional_bias = z['conditional_bias_m'].copy()
        if self.prior.get('joint_RF_auxiliary_selected', False):
            if quality is not None or np.any(np.asarray(self.prior.get('raw_HMM_bias_sigma_m', [0.0] * 4))):
                raise ValueError('Joint RF density uses original IFFT states without a duplicate auxiliary gate or an extra IFFT-only bias')
            if self.prior.get('joint_RF_auxiliary_bias_prior_selected', False) and self.prior.get('joint_RF_family_mixture_selected', False):
                raise ValueError('Auxiliary bias and family-mixture experiments are separate branches')
            with np.load(Path(self.prior['joint_RF_auxiliary_query_root']) / args[0].parent.name / args[0].name / 'CS_history_queries.npz', allow_pickle=False) as z:
                if not np.array_equal(z['anchor_index'], self.anchor_index):
                    raise ValueError('Joint RF auxiliary anchor mismatch')
                if not np.allclose((z['timestamp_ns'] - self.s['start_ns']) / 1000000000.0, self.ct, rtol=0, atol=1e-08):
                    raise ValueError('Joint RF auxiliary clock mismatch')
                values = z['auxiliary_values'][:, :2].copy()
                valid = z['auxiliary_is_finite'][:, :2] & (values > 0)
                valid[:, 1] &= z['auxiliary_values'][:, 3] > 0
                auxiliary = (values, valid)
        for anchor in range(4):
            ix = np.where(self.anchor_index == anchor)[0]
            if not len(ix):
                continue
            context = block_context(anchor, self.ct[ix], self.raw_range[ix], raw_scale(self.ct[ix], self.raw_range[ix]), self.prior['raw_HMM_parameters'], self.prior['raw_HMM_background_range_m'], self.prior['raw_HMM_temporal_selected'], None if quality is None else quality[ix], None if conditional_bias is None else conditional_bias[ix])
            if auxiliary is not None:
                context.update(auxiliary_parameters=self.prior['joint_RF_auxiliary_parameters'], auxiliary_values=auxiliary[0][ix], auxiliary_valid=auxiliary[1][ix])
                if self.prior.get('joint_RF_family_mixture_selected', False):
                    from trajectory_joint_RF import null_auxiliary_energy
                    context['auxiliary_null_parameters'] = self.prior['joint_RF_null_auxiliary_parameters']
                    context['auxiliary_null_loss'] = null_auxiliary_energy(context)
                    context['auxiliary_family_probability'] = self.prior['joint_RF_family_probability'][anchor]
            self.hmm.append((ix, context))

    def auxiliary_bias(self, latent):
        if not self.prior.get('joint_RF_auxiliary_bias_prior_selected', False):
            return (None, None)
        standard = np.broadcast_to(np.asarray(self.prior.get('path_CS_bias_sigma_m', 0.5), float), (4,))[:2]
        L = np.asarray(self.prior['joint_RF_auxiliary_bias_prior_cholesky_m'], float)
        return (np.asarray(self.prior['joint_RF_auxiliary_bias_prior_mean_m']) + L @ (latent[:2] / standard), L / standard[None])

    def path_objective(self, vector, mode):
        if self.prior.get('direct_CS_policy') == 'joint_raw_corrected':
            return DualAxialPath.path_objective(self, vector, mode)
        value, gradient = SensorPath.objective(self, vector, mode)
        q, latent = self.unpack(vector)
        delta = self.C @ q - self.raw_anchor
        distance = np.sqrt(np.square(delta).sum(-1) + 1e-08)
        bias_sigma = np.asarray(self.prior.get('raw_HMM_bias_sigma_m', [0.0] * 4), float)
        bias = bias_sigma * latent
        auxiliary_bias, auxiliary_map = self.auxiliary_bias(latent)
        gd = np.zeros(len(distance))
        weight = self.prior.get('trajectory_factor_weights', {}).get('range', 1.0)
        for ix, context in self.hmm:
            anchor = context['anchor']
            if auxiliary_bias is not None:
                context['auxiliary_bias_m'] = auxiliary_bias
            loss, g, _ = raw_hmm_energy(distance[ix], context, True, float(bias[anchor]))
            value += loss * weight
            gd[ix] = g * weight
            gradient[-4 + anchor] += float(g.sum() / context['parameters']['gain']) * bias_sigma[anchor] * weight
            if auxiliary_bias is not None:
                gradient[-4:-2] += auxiliary_map.T @ context['last_auxiliary_bias_gradient'] * weight
        g = self.C.T @ (gd[:, None] * delta / distance[:, None])
        gradient[:2 * (self.n - 1)] += g[1:].ravel()
        return (value, gradient)

    def solve_axis(self, alternative, modes):
        predictions, diagnostic = super().solve_axis(alternative, modes)
        if self.prior.get('direct_CS_policy') == 'joint_raw_corrected':
            diagnostic.update(direct_CS_policy='joint_raw_corrected', raw_HMM_factor_used=False, corrected_CS_factor='Frozen preprocessed range with source-fitted temporal response and shared raw noise')
            return (predictions, diagnostic)
        q, latent = self.unpack(self.last_selected_vector[:-1])
        distance = np.linalg.norm(self.C @ q - self.raw_anchor, axis=-1)
        bias = np.asarray(self.prior.get('raw_HMM_bias_sigma_m', [0.0] * 4)) * latent
        auxiliary_bias, _ = self.auxiliary_bias(latent)
        posterior = []
        for ix, context in self.hmm:
            if auxiliary_bias is not None:
                context['auxiliary_bias_m'] = auxiliary_bias
            _, _, gamma = raw_hmm_energy(distance[ix], context, True, float(bias[context['anchor']]))
            posterior.extend(gamma)
        if posterior:
            posterior = np.asarray(posterior)
            diagnostic['raw_HMM_mean_component_posterior'] = posterior.mean(0).tolist()
            diagnostic['raw_HMM_high_background_fraction'] = float(np.mean(posterior[:, -1] > 0.5))
            diagnostic['raw_HMM_inferred_anchor_offsets_m'] = bias.tolist()
            if auxiliary_bias is not None:
                diagnostic['joint_RF_auxiliary_inferred_bias_m'] = auxiliary_bias.tolist()
            if self.prior.get('joint_RF_family_mixture_selected', False):
                diagnostic['joint_RF_geometry_family_posterior'] = [dict(anchor=int(context['anchor']), probability=float(context['last_auxiliary_family_posterior'])) for _, context in self.hmm]
        return (predictions, diagnostic)

class ForceRawHMMPath(ForceDualPath, RawHMMPath):
    pass

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    fit(p.parse_args())
if __name__ == '__main__':
    main()
