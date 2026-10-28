import argparse
from functools import lru_cache
import fcntl
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.special import softmax
import torch
from torch import nn
from torch.nn import functional as F
from engine_utils import write
from trajectory_axial_motion import feature_speeds
from trajectory_velocity_states import CONST, source_rows, velocity_energy

def tokens(evidence, times):
    index = np.searchsorted(evidence['times'], times, side='right') - 1
    available = index >= 0
    context = np.asarray(evidence['motion'])[index.clip(0)].copy()
    context[~available] = 0.0
    return np.c_[context, available.astype(float)]

class MotionTCN(nn.Module):

    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv1d(26, 12, 3)
        self.c1 = nn.Conv1d(12, 8, 3, dilation=2)
        self.head = nn.Linear(8, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = F.silu(self.c0(F.pad(x.transpose(1, 2), (2, 0))))
        h = F.silu(self.c1(F.pad(h, (4, 0))))
        return torch.tanh(self.head(h.transpose(1, 2))) * x.new_tensor([0.75, 0.75, 0.75, 0.5])

def numpy_forward(x, model):
    value = (np.asarray(x, float) - np.asarray(model['normalization_mean'])) / np.asarray(model['normalization_scale'])
    value = value.clip(-8, 8)
    weights = model['state']
    h = value
    for name, dilation in [('c0', 1), ('c1', 2)]:
        W = np.asarray(weights[name + '.weight'])
        b = np.asarray(weights[name + '.bias'])
        padded = np.pad(h, ((2 * dilation, 0), (0, 0)))
        h = sum((padded[j * dilation:j * dilation + len(h)] @ W[:, :, j].T for j in range(3))) + b
        h = h / (1 + np.exp(-h.clip(-700, 700)))
    return np.tanh(h @ np.asarray(weights['head.weight']).T + np.asarray(weights['head.bias'])) * np.asarray([0.75, 0.75, 0.75, 0.5])

@lru_cache(maxsize=4)
def loaded(model_file, features):
    return (json.loads(Path(model_file).read_text()), feature_speeds(Path(features)))

def context(prior, cohort, tid, mid, speed, dt, speed_sigma=None):
    model, evidence = loaded(prior['IMU_motion_context_model'], prior['IMU_motion_context_features'])
    x = tokens(evidence[cohort, tid], mid)
    effect = numpy_forward(x, model) * float(prior.get('IMU_motion_context_strength', 1.0))
    p = np.asarray(prior['velocity_state_parameters'])
    N = len(mid)
    gain = math.exp(p[3])
    new_speed = np.asarray(speed) * np.exp(effect[:, 3])
    means = np.zeros((N, 5, 2))
    means[:, 0, 0] = new_speed
    means[:, 1, 0] = -new_speed
    means[:, 2, 1] = gain * new_speed
    means[:, 3, 1] = -gain * new_speed
    sp, st, ss = np.exp(p[:3])
    sigma = np.asarray([[sp, st], [sp, st], [st, sp], [st, sp], [ss, ss]])[None] / np.sqrt(dt)[:, None, None]
    if len(p) > 9:
        extra = np.zeros_like(sigma)
        variance = math.exp(p[9]) * np.square(speed_sigma)
        extra[:, 0:2, 0] = variance[:, None]
        extra[:, 2:4, 1] = gain * gain * variance[:, None]
        sigma = np.sqrt(sigma * sigma + extra)
    mass = softmax([0.0, p[4], p[5]])
    prob = np.asarray([mass[0] / 2, mass[0] / 2, mass[1] / 2, mass[1] / 2, mass[2]])
    persistence = np.exp(p[[6, 6, 7, 7, 8]])
    transition = []
    for i, gap in enumerate(0.5 * (dt[:-1] + dt[1:])):
        rate = np.exp(effect[i + 1, [0, 0, 1, 1, 2]])
        stick = np.exp(-gap * rate / persistence) if prior['velocity_state_temporal_selected'] else np.zeros(5)
        transition.append((1 - stick[:, None]) * prob[None] + np.diag(stick))
    return dict(means=means, sigma=sigma, probability=prob, transition=transition, effects=effect)

def energy(velocity, ctx, derivative=False):
    e = np.asarray(velocity)[:, None] - ctx['means']
    sigma = ctx['sigma']
    emission = (CONST - np.log(sigma) - 2 * np.log1p(e * e / (3 * sigma * sigma))).sum(-1)
    maximum = emission.max(-1)
    density = np.exp(emission - maximum[:, None])
    N = len(e)
    alpha = np.empty_like(density)
    normalizer = np.empty(N)
    alpha[0] = ctx['probability'] * density[0]
    normalizer[0] = alpha[0].sum()
    alpha[0] /= normalizer[0]
    for i in range(1, N):
        alpha[i] = alpha[i - 1] @ ctx['transition'][i - 1] * density[i]
        normalizer[i] = alpha[i].sum()
        alpha[i] /= normalizer[i]
    value = -float((np.log(normalizer) + maximum).sum())
    if not derivative:
        return value
    beta = np.ones_like(alpha)
    for i in range(N - 2, -1, -1):
        beta[i] = ctx['transition'][i] @ (density[i + 1] * beta[i + 1]) / normalizer[i + 1]
    posterior = alpha * beta
    posterior /= posterior.sum(-1, keepdims=True)
    gradient = (posterior[:, :, None] * 4 * e / (3 * sigma * sigma + e * e)).sum(1)
    return (value, gradient, posterior)

def train(a):
    torch.set_num_threads(1)
    torch.manual_seed(80620)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    rows = source_rows(a)
    evidence = feature_speeds(a.features)
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    phone = {s['trajectory_id']: s['device'] for s in segments}
    B = len(rows)
    M = max((len(r['full_dt']) for r in rows))
    x = np.zeros((B, M, 26))
    v = np.zeros((B, M, 2))
    speed = np.zeros((B, M))
    sigma_e = np.ones((B, M))
    dt = np.ones((B, M)) * 0.5
    valid = np.zeros((B, M), bool)
    length = np.zeros((B, M), bool)
    for i, r in enumerate(rows):
        n = len(r['full_dt'])
        x[i, :n] = tokens(evidence['source_train', r['trajectory_id']], r['full_time_s'])
        v[i, r['valid_indices']] = r['velocity']
        speed[i, :n] = r['full_speed']
        sigma_e[i, :n] = r['full_speed_sigma']
        dt[i, :n] = r['full_dt']
        valid[i, r['valid_indices']] = True
        length[i, :n] = True
    held = np.asarray([r['heldout'] for r in rows])
    train_mask = ~held
    mean = x[train_mask][length[train_mask]].mean(0)
    scale = x[train_mask][length[train_mask]].std(0).clip(0.1)
    inputs = np.clip((x - mean) / scale, -8, 8)
    device = 'cuda:0'
    tensor = lambda y, dtype=torch.float32: torch.tensor(y, device=device, dtype=dtype)
    inp = tensor(inputs)
    vel = tensor(v)
    sp = tensor(speed)
    step = tensor(dt)
    se = tensor(sigma_e)
    observed = tensor(valid, torch.bool)
    exists = tensor(length, torch.bool)
    p0 = tensor(prior['velocity_state_parameters'])
    state_group = torch.tensor([0, 0, 1, 1, 2], device=device)
    mass = torch.softmax(torch.stack((p0[4] * 0, p0[4], p0[5])), 0)
    pi = torch.stack((mass[0] / 2, mass[0] / 2, mass[1] / 2, mass[1] / 2, mass[2]))
    tau = p0[torch.tensor([6, 6, 7, 7, 8], device=device)].exp()
    ss1, ss2, ss0 = p0[:3].exp()
    sig0 = torch.stack((torch.stack((ss1, ss2)), torch.stack((ss1, ss2)), torch.stack((ss2, ss1)), torch.stack((ss2, ss1)), torch.stack((ss0, ss0))))
    sig = sig0[None, None] / step[:, :, None, None].sqrt()
    gain = p0[3].exp()
    if len(p0) > 9:
        z = torch.zeros_like(se)
        var = p0[9].exp() * se.square()
        extra = torch.stack((torch.stack((var, z), -1), torch.stack((var, z), -1), torch.stack((z, gain.square() * var), -1), torch.stack((z, gain.square() * var), -1), torch.stack((z, z), -1)), 2)
        sig = (sig.square() + extra).sqrt()
    model = MotionTCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    group_weights = np.zeros(B)
    for ph in sorted(set(phone.values())):
        ix = np.asarray([phone[r['trajectory_id']] == ph and (not r['heldout']) for r in rows])
        group_weights[ix] = 1 / max(valid[ix].sum(), 1) / 4
    weights = tensor(group_weights)
    held_t = tensor(held, torch.bool)

    def losses(effect):
        s = sp * effect[:, :, 3].exp()
        z = torch.zeros_like(s)
        means = torch.stack((torch.stack((s, z), -1), torch.stack((-s, z), -1), torch.stack((z, gain * s), -1), torch.stack((z, -gain * s), -1), torch.stack((z, z), -1)), 2)
        e = vel[:, :, None] - means
        emission = (CONST - sig.log() - 2 * torch.log1p(e.square() / (3 * sig.square()))).sum(-1)
        emission = torch.where(observed[:, :, None], emission, torch.zeros_like(emission))
        alpha = pi.log()[None] + emission[:, 0]
        for j in range(1, M):
            gap = 0.5 * (step[:, j] + step[:, j - 1])
            stick = torch.exp(-gap[:, None] * effect[:, j, state_group].exp() / tau[None])
            T = (1 - stick[:, :, None]) * pi[None, None] + torch.diag_embed(stick)
            proposed = torch.logsumexp(alpha[:, :, None] + T.clamp_min(1e-15).log(), 1) + emission[:, j]
            alpha = torch.where(exists[:, j, None], proposed, alpha)
        return -torch.logsumexp(alpha, -1)
    trace = []
    best = None
    baseline = None
    for update in range(a.steps + 1):
        effect = model(inp)
        values = losses(effect)
        if update % 20 == 0 or update == a.steps:
            heldval = float(values[held_t].sum() / observed[held_t].sum())
            record = dict(update=update, source_train_phone_balanced_NLL=float((values * weights).sum()), source_heldout_NLL=heldval)
            trace.append(record)
            if baseline is None:
                baseline = heldval
            if best is None or heldval < best[0]:
                best = (heldval, update, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if update == a.steps:
            break
        train_exists = exists & ~held_t[:, None]
        regularizer = (effect.square() * train_exists[:, :, None]).sum() / train_exists.sum() * 0.003
        objective = (values * weights).sum() + regularizer
        optimizer.zero_grad()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
    model.load_state_dict(best[2])
    saved = dict(state={k: v.numpy().tolist() for k, v in best[2].items()}, normalization_mean=mean.tolist(), normalization_scale=scale.tolist(), parameters=sum((p.numel() for p in model.parameters())), input='Existing25 causal IMU scalars plus availability mask; no phone bearing, identity or position', network='Conv26to12k3d1 SiLU;Conv12to8k3d2 SiLU;Linear8to4 tanh; bounded log rate3 and speed gain1', selected_update=best[1], source_train_paths=43, source_heldout_paths=11, source_baseline_heldout_NLL=baseline, source_selected_heldout_NLL=best[0], source_heldout_preferred=best[0] < baseline - 1e-06, batch_size=128, actual_source_batch=B, trace=trace, target_GT_used=False, seconds=time.monotonic() - began)
    test_input = x[0, :int(length[0].sum())]
    expected = model(tensor(inputs[0, :len(test_input)])[None])[0].detach().cpu().numpy()
    actual = numpy_forward(test_input, saved)
    difference = float(abs(actual - expected).max())
    if difference > 3e-06:
        raise ValueError('CPU motion TCN differs from GPU training')
    changed = test_input.copy()
    prefix = len(changed) // 2
    changed[prefix:] += 100
    prefixdiff = float(abs(numpy_forward(changed, saved)[:prefix] - actual[:prefix]).max())
    if prefixdiff != 0:
        raise ValueError('Future IMU changes affect an earlier motion prediction')
    saved.update(CPU_GPU_maximum_difference=difference, future_perturbation_prefix_difference=prefixdiff)
    a.output.mkdir(parents=True, exist_ok=True)
    write(a.output / 'model.json', saved)
    prior.update(IMU_motion_context_model=str((a.output / 'model.json').resolve()), IMU_motion_context_features=str(a.features.resolve()), IMU_motion_context_source_fitted=True, IMU_motion_context_target_GT_used=False, cache_solver_groups=True, save_solver_state=True)
    row = rows[0]
    ctx = context(prior, 'source_train', row['trajectory_id'], row['full_time_s'], row['full_speed'], row['full_dt'], row['full_speed_sigma'])
    idx = row['valid_indices']
    selected = dict(means=ctx['means'][idx], sigma=ctx['sigma'][idx], probability=ctx['probability'], transition=[ctx['transition'][int(j) - 1] for j in idx[1:]])
    value, g, _ = energy(row['velocity'], selected, True)
    direction = np.random.RandomState(80620).normal(size=g.shape)
    direction /= np.linalg.norm(direction)
    h = 1e-05
    numeric = (energy(row['velocity'] + h * direction, selected) - energy(row['velocity'] - h * direction, selected)) / (2 * h)
    gradient_error = float(abs(numeric - (direction * g).sum()))
    if gradient_error > 1e-05:
        raise ValueError('Conditional motion gradient differs from native objective')
    saved['native_velocity_gradient_difference'] = gradient_error
    write(a.output / 'model.json', saved)
    write(a.output / 'source_prior.json', prior)
    zero = context(dict(prior, IMU_motion_context_strength=0.0), 'source_train', row['trajectory_id'], row['time_s'], row['speed'], row['dt'], row['speed_sigma'])
    zero_result = energy(row['velocity'], zero, True)
    old = velocity_energy(row['velocity'], row['speed'], row['dt'], prior['velocity_state_parameters'], prior['velocity_state_temporal_selected'], True, row['speed_sigma'])
    zero_difference = max(abs(zero_result[0] - old[0]), float(abs(zero_result[1] - old[1]).max()))
    if zero_difference > 1e-10:
        raise ValueError('Zero context does not reproduce old motion model')
    saved['zero_context_old_model_difference'] = zero_difference
    write(a.output / 'model.json', saved)
    print(json.dumps({k: saved[k] for k in ['parameters', 'selected_update', 'source_baseline_heldout_NLL', 'source_selected_heldout_NLL', 'source_heldout_preferred', 'seconds', 'CPU_GPU_maximum_difference', 'native_velocity_gradient_difference']}), flush=True)

def main():
    p = argparse.ArgumentParser()
    for k in ['base', 'capture', 'features', 'prior', 'output']:
        p.add_argument('--' + k, type=Path, required=True)
    p.add_argument('--steps', type=int, default=400)
    a = p.parse_args()
    with (a.base / 'research_execution.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(a)
if __name__ == '__main__':
    main()
