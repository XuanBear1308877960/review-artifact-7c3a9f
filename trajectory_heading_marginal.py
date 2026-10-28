import math
import numpy as np
from scipy.special import softmax, logsumexp, i0e
from trajectory_velocity_states import CONST

def context(speed, dt, parameters, axis_parameters, speed_sigma=None, hypotheses=256):
    p = np.asarray(parameters)
    dt = np.asarray(dt)
    N = len(speed)
    angle = (np.arange(hypotheses) + 0.5) * math.pi / hypotheses
    u = np.stack((np.cos(angle), np.sin(angle)), -1)
    v = np.stack((-np.sin(angle), np.cos(angle)), -1)
    ap = np.asarray(axis_parameters)
    kappa = np.exp(ap[2:4])
    mix = 1 / (1 + np.exp(-ap[4]))
    logs = np.log([mix, 1 - mix])[None] + kappa[None] * np.cos(2 * (angle[:, None] - ap[None, :2])) - math.log(math.pi) - np.log(i0e(kappa))[None] - kappa[None]
    log_prior = logsumexp(logs, -1)
    log_prior -= logsumexp(log_prior)
    means = np.zeros((N, 5, 2))
    gain = math.exp(p[3])
    means[:, 0, 0] = speed
    means[:, 1, 0] = -speed
    means[:, 2, 1] = gain * speed
    means[:, 3, 1] = -gain * speed
    sp, st, ss = np.exp(p[:3])
    sigma = np.asarray([[sp, st], [sp, st], [st, sp], [st, sp], [ss, ss]])[None] / np.sqrt(dt)[:, None, None]
    if len(p) > 9:
        if speed_sigma is None:
            raise ValueError('Missing measured speed uncertainty')
        extra = np.zeros_like(sigma)
        variance = np.exp(p[9]) * np.square(speed_sigma)
        extra[:, 0:2, 0] = variance[:, None]
        extra[:, 2:4, 1] = gain * gain * variance[:, None]
        sigma = np.sqrt(sigma * sigma + extra)
    mass = softmax([0.0, p[4], p[5]])
    prob = np.asarray([mass[0] / 2, mass[0] / 2, mass[1] / 2, mass[1] / 2, mass[2]])
    tau = np.exp(p[[6, 6, 7, 7, 8]])
    transition = []
    for gap in 0.5 * (dt[1:] + dt[:-1]):
        stick = np.exp(-gap / tau)
        transition.append((1 - stick[:, None]) * prob[None] + np.diag(stick))
    return dict(u=u, v=v, means=means, sigma=sigma, probability=prob, transitions=transition, log_prior=log_prior, angle=angle)

def energy(velocity, ctx, derivative=True):
    vp = ctx['u'] @ np.asarray(velocity).T
    vt = ctx['v'] @ np.asarray(velocity).T
    residual = np.stack((vp, vt), -1)[:, :, None] - ctx['means'][None]
    sig = ctx['sigma'][None]
    emission = (CONST - np.log(sig) - 2 * np.log1p(residual ** 2 / (3 * sig ** 2))).sum(-1)
    P, N, K = emission.shape
    maximum = emission.max(-1)
    density = np.exp(emission - maximum[:, :, None])
    alpha = np.empty_like(density)
    scale = np.empty((P, N))
    alpha[:, 0] = ctx['probability'][None] * density[:, 0]
    scale[:, 0] = alpha[:, 0].sum(-1)
    alpha[:, 0] /= scale[:, 0, None]
    for i in range(1, N):
        alpha[:, i] = alpha[:, i - 1] @ ctx['transitions'][i - 1] * density[:, i]
        scale[:, i] = alpha[:, i].sum(-1)
        alpha[:, i] /= scale[:, i, None]
    conditional_energy = -(np.log(scale) + maximum).sum(-1)
    evidence = ctx['log_prior'] - conditional_energy
    normalizer = logsumexp(evidence)
    mass = np.exp(evidence - normalizer)
    if not derivative:
        return -float(normalizer)
    backward = np.ones_like(alpha)
    for i in range(N - 2, -1, -1):
        backward[:, i] = density[:, i + 1] * backward[:, i + 1] @ ctx['transitions'][i].T / scale[:, i + 1, None]
    phase = alpha * backward
    phase /= phase.sum(-1, keepdims=True)
    score = (phase[:, :, :, None] * 4 * residual / (3 * sig ** 2 + residual ** 2)).sum(2)
    world = score[..., 0, None] * ctx['u'][:, None] + score[..., 1, None] * ctx['v'][:, None]
    return (-float(normalizer), (mass[:, None, None] * world).sum(0), mass)
