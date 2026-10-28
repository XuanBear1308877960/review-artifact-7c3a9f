import argparse
import fcntl
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import expm
from scipy.special import softmax
import torch
from engine_utils import write
from trajectory_velocity_states import CONST, source_rows
from trajectory_IMU_motion import energy

def generator(parameters, phases):
    p = np.asarray(parameters)
    mass = softmax([0.0, p[3], p[4]])
    pi = np.asarray([mass[0] / 2, mass[0] / 2, mass[1] / 2, mass[1] / 2, mass[2]])
    rate = phases / np.exp(p[[0, 0, 1, 1, 2]])
    Q = np.zeros((5 * phases, 5 * phases))
    for state in range(5):
        for phase in range(phases):
            row = state * phases + phase
            if phase < phases - 1:
                Q[row, row + 1] = rate[state]
            else:
                Q[row, np.arange(5) * phases] = rate[state] * pi
    Q -= np.diag(Q.sum(-1))
    return (Q, np.repeat(pi / phases, phases))

def context(prior, speed, dt, speed_sigma=None):
    phases = int(prior['duration_motion_phases'])
    Q, pi = generator(prior['duration_motion_parameters'], phases)
    p = np.asarray(prior['velocity_state_parameters'])
    means = np.zeros((len(speed), 5, 2))
    means[:, 0, 0] = speed
    means[:, 1, 0] = -np.asarray(speed)
    means[:, 2, 1] = math.exp(p[3]) * np.asarray(speed)
    means[:, 3, 1] = -math.exp(p[3]) * np.asarray(speed)
    sp, st, ss = np.exp(p[:3])
    sigma = np.asarray([[sp, st], [sp, st], [st, sp], [st, sp], [ss, ss]])[None] / np.sqrt(dt)[:, None, None]
    if len(p) > 9:
        extra = np.zeros_like(sigma)
        v = math.exp(p[9]) * np.square(speed_sigma)
        extra[:, :2, 0] = v[:, None]
        extra[:, 2:4, 1] = math.exp(2 * p[3]) * v[:, None]
        sigma = np.sqrt(sigma * sigma + extra)
    transitions = []
    cache = {}
    for gap in 0.5 * (np.asarray(dt[:-1]) + np.asarray(dt[1:])):
        key = round(float(gap), 10)
        if key not in cache:
            T = expm(Q * gap)
            if T.min() < -1e-12 or abs(T.sum(-1) - 1).max() > 1e-10:
                raise ValueError('Invalid duration transition matrix')
            cache[key] = T.clip(0)
        transitions.append(cache[key])
    return dict(means=np.repeat(means, phases, axis=1), sigma=np.repeat(sigma, phases, axis=1), probability=pi, transition=transitions)

def train(a):
    torch.set_num_threads(1)
    torch.manual_seed(80621)
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    rows = source_rows(a)
    B = len(rows)
    M = max((len(r['full_dt']) for r in rows))
    velocity = np.zeros((B, M, 2))
    speed = np.zeros((B, M))
    dt = np.ones((B, M)) * 0.5
    valid = np.zeros((B, M), bool)
    exists = np.zeros((B, M), bool)
    for i, r in enumerate(rows):
        n = len(r['full_dt'])
        velocity[i, r['valid_indices']] = r['velocity']
        speed[i, :n] = r['full_speed']
        dt[i, :n] = r['full_dt']
        valid[i, r['valid_indices']] = True
        exists[i, :n] = True
    device = 'cuda:0'
    tensor = lambda v, dtype=torch.float64: torch.tensor(v, device=device, dtype=dtype)
    vel, sp, step = map(tensor, [velocity, speed, dt])
    observed = tensor(valid, torch.bool)
    length = tensor(exists, torch.bool)
    held = tensor([r['heldout'] for r in rows], torch.bool)
    p = np.asarray(prior['velocity_state_parameters'])
    s1, s2, ss = np.exp(p[:3])
    sigma = tensor([[s1, s2], [s1, s2], [s2, s1], [s2, s1], [ss, ss]])[None, None] / step[:, :, None, None].sqrt()
    if len(p) > 9:
        raise ValueError('Duration fit version requires the selected9-parameter motion baseline')
    zero = torch.zeros_like(sp)
    gain = math.exp(p[3])
    means = torch.stack((torch.stack((sp, zero), -1), torch.stack((-sp, zero), -1), torch.stack((zero, gain * sp), -1), torch.stack((zero, -gain * sp), -1), torch.stack((zero, zero), -1)), 2)
    e = vel[:, :, None] - means
    emission = (CONST - sigma.log() - 2 * torch.log1p(e.square() / (3 * sigma.square()))).sum(-1)
    emission = torch.where(observed[:, :, None], emission, torch.zeros_like(emission))
    gap = tensor(np.asarray([float(r['full_dt'][0]) for r in rows]))
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    phones = {s['trajectory_id']: s['device'] for s in segments}
    weights = np.zeros(B)
    for ph in sorted(set(phones.values())):
        ix = np.asarray([phones[r['trajectory_id']] == ph and (not r['heldout']) for r in rows])
        weights[ix] = 1 / max(valid[ix].sum(), 1) / 4
    weights = tensor(weights)
    initial = np.r_[p[6:9], p[4:6]]
    reports = []
    global_best = None
    for phases in [1, 2, 4]:
        K = 5 * phases
        source = []
        destination = []
        coefficient = []
        for state in range(5):
            for phase in range(phases):
                if phase < phases - 1:
                    source.append(state * phases + phase)
                    destination.append(state * phases + phase + 1)
                    coefficient.append(-1)
                else:
                    for other in range(5):
                        source.append(state * phases + phase)
                        destination.append(other * phases)
                        coefficient.append(other)
        src = torch.tensor(source, device=device)
        dst = torch.tensor(destination, device=device)
        dest = torch.tensor(coefficient, device=device)
        groups = src // phases
        parameter = torch.nn.Parameter(tensor(initial))
        opt = torch.optim.Adam([parameter], lr=0.025)
        best = None
        trace = []
        E = emission.repeat_interleave(phases, dim=-1)

        def losses():
            mass = torch.softmax(torch.stack((parameter[3] * 0, parameter[3], parameter[4])), 0)
            pi = torch.stack((mass[0] / 2, mass[0] / 2, mass[1] / 2, mass[1] / 2, mass[2]))
            rate = phases / parameter[torch.tensor([0, 0, 1, 1, 2], device=device)].exp()
            values = rate[groups] * torch.where(dest < 0, torch.ones_like(dest, dtype=torch.float64), pi[dest.clamp_min(0)])
            Q = torch.zeros((K, K), device=device, dtype=torch.float64)
            Q.index_put_((src, dst), values, accumulate=True)
            Q = Q - torch.diag(Q.sum(-1))
            T = torch.matrix_exp(gap[:, None, None] * Q[None]).clamp_min(1e-30).log()
            alpha = (pi.repeat_interleave(phases) / phases).log()[None] + E[:, 0]
            for j in range(1, M):
                proposed = torch.logsumexp(alpha[:, :, None] + T, 1) + E[:, j]
                alpha = torch.where(length[:, j, None], proposed, alpha)
            return -torch.logsumexp(alpha, -1)
        for update in range(a.steps + 1):
            values = losses()
            if update % 20 == 0 or update == a.steps:
                validation = float(values[held].sum() / observed[held].sum())
                trace.append(dict(update=update, source_train_phone_balanced_NLL=float((values * weights).sum()), source_heldout_NLL=validation))
                if best is None or validation < best[0]:
                    best = (validation, update, parameter.detach().cpu().numpy().copy())
            if update == a.steps:
                break
            loss = (values * weights).sum() + 0.002 * (parameter - tensor(initial)).square().sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter], 10.0, error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                parameter[:3].clamp_(math.log(0.04), math.log(30))
                parameter[3:].clamp_(-6.0, 3.0)
        result = dict(phases=phases, expanded_states=K, source_heldout_NLL=best[0], selected_update=best[1], parameters=best[2].tolist(), trace=trace)
        reports.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != 'trace'}), flush=True)
        if global_best is None or best[0] < global_best['source_heldout_NLL']:
            global_best = result
    selected = dict(prior, duration_motion_phases=global_best['phases'], duration_motion_parameters=global_best['parameters'], duration_motion_source_heldout_NLL=global_best['source_heldout_NLL'], duration_motion_source_train43_heldout11=True, duration_motion_target_GT_used=False, save_solver_state=True, cache_solver_groups=True)
    r = rows[0]
    ctx = context(selected, r['speed'], r['dt'], r['speed_sigma'])
    value, g, _ = energy(r['velocity'], ctx, True)
    v = np.random.RandomState(80621).normal(size=g.shape)
    v /= np.linalg.norm(v)
    h = 1e-05
    numeric = (energy(r['velocity'] + h * v, ctx) - energy(r['velocity'] - h * v, ctx)) / (2 * h)
    difference = float(abs(numeric - (g * v).sum()))
    if difference > 1e-05:
        raise ValueError('Duration native velocity gradient mismatch')
    a.output.mkdir(parents=True, exist_ok=True)
    write(a.output / 'source_prior.json', selected)
    write(a.output / 'source_fit.json', dict(status='COMPLETE_SOURCE_FITTED_DURATION_MOTION', reports=reports, selected=global_best, train_paths=43, selection_paths=11, new_neural_parameters=0, new_fitted_parameters=5, old_velocity_means_and_scales_unchanged=True, source_reference_length_not_input=True, phone_bearing_not_input=True, initial_phase_unknown_uniform=True, native_velocity_gradient_difference=difference, seconds=time.monotonic() - began))
    print(json.dumps(dict(selected_phases=global_best['phases'], source_heldout_NLL=global_best['source_heldout_NLL'], gradient_difference=difference, seconds=time.monotonic() - began)), flush=True)

def main():
    p = argparse.ArgumentParser()
    for k in ['base', 'capture', 'features', 'prior', 'output']:
        p.add_argument('--' + k, type=Path, required=True)
    p.add_argument('--steps', type=int, default=200)
    a = p.parse_args()
    with (a.base / 'research_execution.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(a)
if __name__ == '__main__':
    main()
