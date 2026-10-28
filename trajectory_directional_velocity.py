import argparse
import fcntl
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import expm
from scipy.special import expit, softmax, logsumexp
import torch
from engine_utils import write
from trajectory_velocity_states import CONST, source_rows, velocity_energy

def context(parameters, dt, times=None, directions=16):
    p = np.asarray(parameters, float)
    dt = np.asarray(dt, float)
    angle = np.arange(directions) * 2 * math.pi / directions
    u = np.stack((np.cos(angle), np.sin(angle)), -1)
    v = np.stack((-np.sin(angle), np.cos(angle)), -1)
    angular = softmax(p[5] * np.cos(2 * angle) + p[6] * np.cos(4 * angle))
    stop = expit(p[7])
    prob = np.r_[(1 - stop) * angular, stop]
    cosine = np.cos(angle[:, None] - angle[None])
    kappa = np.exp(p[10])
    switch = logsumexp(np.stack((kappa * cosine, p[9] - kappa * cosine), -1), -1) + np.log(angular)[None]
    np.fill_diagonal(switch, -np.inf)
    switch = softmax(switch, axis=-1)
    Q = np.zeros((directions + 1, directions + 1))
    Q[:-1, :-1] = np.exp(p[8]) * switch
    Q[:-1, -1] = np.exp(p[11])
    Q[-1, :-1] = np.exp(p[12]) * angular
    np.fill_diagonal(Q, -Q.sum(-1))
    mid = np.cumsum(dt) - 0.5 * dt if times is None else np.asarray(times, float)
    transitions = [expm(Q * gap) for gap in np.diff(mid)]
    if transitions and (min((float(T.min()) for T in transitions)) < -1e-12 or max((float(abs(T.sum(-1) - 1).max()) for T in transitions)) > 1e-10):
        raise ValueError('Directional motion transition is not a probability matrix')
    sigma = np.exp(p[:2])[None, None] / np.sqrt(dt)[:, None, None]
    gain = np.exp(p[3] * np.cos(angle) ** 2 + p[4] * np.sin(angle) ** 2)
    return dict(u=u, v=v, gain=gain, sigma=sigma, stop_sigma=np.exp(p[2]) / np.sqrt(dt), probability=prob, transitions=transitions)

def energy(velocity, speed, ctx, derivative=False):
    velocity = np.asarray(velocity)
    speed = np.asarray(speed)
    N = len(speed)
    K = len(ctx['probability'])
    residual = np.stack((velocity @ ctx['u'].T - speed[:, None] * ctx['gain'][None], velocity @ ctx['v'].T), -1)
    sig = ctx['sigma']
    moving = (CONST - np.log(sig) - 2 * np.log1p(residual ** 2 / (3 * sig ** 2))).sum(-1)
    stop = ctx['stop_sigma'][:, None]
    still = (CONST - np.log(stop) - 2 * np.log1p(velocity ** 2 / (3 * stop ** 2))).sum(-1)
    emission = np.column_stack((moving, still))
    maximum = emission.max(-1)
    density = np.exp(emission - maximum[:, None])
    alpha = np.empty((N, K))
    scales = np.empty(N)
    alpha[0] = ctx['probability'] * density[0]
    scales[0] = alpha[0].sum()
    alpha[0] /= scales[0]
    for i in range(1, N):
        alpha[i] = alpha[i - 1] @ ctx['transitions'][i - 1] * density[i]
        scales[i] = alpha[i].sum()
        alpha[i] /= scales[i]
    value = -float((np.log(scales) + maximum).sum())
    if not derivative:
        return value
    backward = np.ones_like(alpha)
    for i in range(N - 2, -1, -1):
        backward[i] = ctx['transitions'][i] @ (density[i + 1] * backward[i + 1]) / scales[i + 1]
    weight = alpha * backward
    weight /= weight.sum(-1, keepdims=True)
    score = 4 * residual / (3 * sig ** 2 + residual ** 2)
    g = (weight[:, :-1, None] * (score[..., 0, None] * ctx['u'][None] + score[..., 1, None] * ctx['v'][None])).sum(1)
    g += weight[:, -1, None] * 4 * velocity / (3 * stop ** 2 + velocity ** 2)
    return (value, g, weight)

def fit(a):
    began = time.monotonic()
    torch.set_num_threads(1)
    prior = json.loads(a.prior.read_text())
    rows = source_rows(a)
    N = len(rows)
    M = max((len(r['speed']) for r in rows))
    K = a.directions
    device = 'cuda:0'
    velocity = np.zeros((N, M, 2))
    speed = np.zeros((N, M))
    dt = np.ones((N, M)) * 0.5
    gaps = np.ones((N, M - 1)) * 0.5
    mask = np.zeros((N, M), bool)
    for i, r in enumerate(rows):
        n = len(r['speed'])
        velocity[i, :n] = r['velocity']
        speed[i, :n] = r['speed']
        dt[i, :n] = r['dt']
        gaps[i, :n - 1] = np.diff(r['time_s'])
        mask[i, :n] = True
    vel, sp, step, gap = [torch.tensor(x, device=device, dtype=torch.float64) for x in [velocity, speed, dt, gaps]]
    valid = torch.tensor(mask, device=device)
    held = torch.tensor([r['heldout'] for r in rows], device=device)
    angle = torch.arange(K, device=device, dtype=torch.float64) * (2 * math.pi / K)
    u = torch.stack((angle.cos(), angle.sin()), -1)
    v = torch.stack((-angle.sin(), angle.cos()), -1)
    cosine = torch.cos(angle[:, None] - angle[None])
    diagonal = torch.eye(K, device=device, dtype=torch.bool)
    old = np.asarray(prior['velocity_state_parameters'])
    mass = softmax([0.0, old[4], old[5]])
    initial = np.r_[old[:3], 0.0, old[3], max(0.0, -old[4] / 2), 0.5, np.log(mass[2] / (1 - mass[2])), -old[6], np.log(0.5), np.log(4.0), np.log(0.1), np.log(2.0)]
    lower = np.r_[np.log([0.05, 0.05, 0.05, 0.5, 0.3]), 0.0, 0.0, -7.0, np.log(0.02), np.log(0.001), np.log(0.1), np.log(0.01), np.log(0.05)]
    upper = np.r_[np.log([3.0, 3.0, 3.0, 1.5, 1.5]), 5.0, 5.0, 1.0, np.log(6.0), np.log(8.0), np.log(20.0), np.log(3.0), np.log(12.0)]
    initial = np.clip(initial, lower, upper)
    parameter = torch.nn.Parameter(torch.tensor(initial, device=device, dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=0.025)

    def likelihood(p):
        angular = torch.softmax(p[5] * torch.cos(2 * angle) + p[6] * torch.cos(4 * angle), 0)
        stop = p[7].sigmoid()
        prob = torch.cat(((1 - stop) * angular, stop[None]))
        switch = torch.logsumexp(torch.stack((p[10].exp() * cosine, p[9] - p[10].exp() * cosine), -1), -1) + angular.log()[None]
        switch = torch.softmax(switch.masked_fill(diagonal, -torch.inf), -1)
        Q = torch.cat((torch.cat((p[8].exp() * switch, p[11].exp().expand(K, 1)), -1), torch.cat((p[12].exp() * angular, (p[12] * 0)[None]))[None]), 0)
        Q = Q - torch.diag(Q.sum(-1))
        allgaps = gap.reshape(-1)
        unique, inverse = torch.unique(allgaps, return_inverse=True)
        matrices = torch.linalg.matrix_exp(unique[:, None, None] * Q[None])
        T = matrices[inverse].reshape(N, M - 1, K + 1, K + 1)
        gain = torch.exp(p[3] * angle.cos().square() + p[4] * angle.sin().square())
        residual = torch.stack((vel @ u.T - sp[:, :, None] * gain[None, None], vel @ v.T), -1)
        sigma = p[:2].exp()[None, None, None] / step[:, :, None, None].sqrt()
        moving = (CONST - sigma.log() - 2 * torch.log1p(residual.square() / (3 * sigma.square()))).sum(-1)
        stop_sigma = p[2].exp() / step.sqrt()
        still = (CONST - stop_sigma[:, :, None].log() - 2 * torch.log1p(vel.square() / (3 * stop_sigma[:, :, None].square()))).sum(-1)
        emission = torch.cat((moving, still[:, :, None]), -1)
        alpha = prob.log()[None] + emission[:, 0]
        for j in range(1, M):
            proposed = torch.logsumexp(alpha[:, :, None] + T[:, j - 1].clamp_min(1e-30).log(), 1) + emission[:, j]
            alpha = torch.where(valid[:, j, None], proposed, alpha)
        return -torch.logsumexp(alpha, -1)
    best = None
    trace = []
    for update in range(a.updates + 1):
        losses = likelihood(parameter)
        if update % 25 == 0 or update == a.updates:
            val = float(losses[held].sum() / valid[held].sum())
            row = dict(update=update, source_heldout_NLL=val, source_train_NLL=float(losses[~held].sum() / valid[~held].sum()), seconds=time.monotonic() - began)
            trace.append(row)
            print(json.dumps(row), flush=True)
            if best is None or val < best[0]:
                best = (val, parameter.detach().cpu().numpy().copy(), update)
        if update == a.updates:
            break
        loss = losses[~held].sum() / valid[~held].sum()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameter], 10.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(torch.tensor(lower, device=device), torch.tensor(upper, device=device))
    p = best[1]
    r = rows[0]
    ctx = context(p, r['dt'], r['time_s'], K)
    direction = np.random.RandomState(83199).normal(size=r['velocity'].shape)
    direction /= np.linalg.norm(direction)
    actual, g, _ = energy(r['velocity'], r['speed'], ctx, True)
    h = 1e-05
    numeric = (energy(r['velocity'] + h * direction, r['speed'], ctx) - energy(r['velocity'] - h * direction, r['speed'], ctx)) / (2 * h)
    difference = float(abs(numeric - (g * direction).sum()))
    heldout = sum((energy(r['velocity'], r['speed'], context(p, r['dt'], r['time_s'], K)) for r in rows if r['heldout'])) / sum((len(r['speed']) for r in rows if r['heldout']))
    if difference > 1e-05 or abs(heldout - best[0]) > 1e-07:
        raise ValueError('Directional motion fitted/runtime likelihood or derivative differs')
    baseline = sum((velocity_energy(r['velocity'], r['speed'], r['dt'], old, prior['velocity_state_temporal_selected'], speed_sigma=r['speed_sigma']) for r in rows if r['heldout'])) / sum((len(r['speed']) for r in rows if r['heldout']))
    prior.update(directional_velocity_parameters=p.tolist(), directional_velocity_directions=K, directional_velocity_source_heldout_NLL=heldout, directional_velocity_old_five_source_heldout_NLL=baseline, directional_velocity_selected_update=best[2], directional_velocity_source_only=True, directional_velocity_target_labels_used=False, directional_velocity_phone_heading_used=False, directional_velocity_replaces_five_state_density=True, directional_velocity_derivative_difference=difference, directional_velocity_fit_seconds=time.monotonic() - began, directional_velocity_source_fits=trace, cache_solver_groups=True, save_solver_state=True)
    if prior.get('sensor_path_proposal_root'):
        prior['sensor_path_proposal_recompute_parent_objective'] = True
    write(a.output / 'source_prior.json', prior)
    print(json.dumps(dict(status='COMPLETE_SOURCE_CIRCULAR_VELOCITY_DENSITY_NOT_LOCATION_EVIDENCE', directions=K, parameters=len(p), source_heldout_NLL=heldout, old_five_NLL=baseline, derivative_difference=difference, seconds=time.monotonic() - began)), flush=True)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'features', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--directions', type=int, default=16)
    p.add_argument('--updates', type=int, default=200)
    a = p.parse_args()
    with (a.base / 'research_execution.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fit(a)
if __name__ == '__main__':
    main()
