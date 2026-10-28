import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from engine_utils import write
from trajectory_radio_gp import SpatialRadioField

def metric(parameters):
    a, theta = parameters[:2]
    c, s = (math.cos(theta), math.sin(theta))
    R = np.array([[c, -s], [s, c]])
    dR = np.array([[-s, -c], [c, -s]])
    diagonal = np.diag([math.exp(a), math.exp(-a)])
    G = R @ diagonal @ R.T
    d_a = R @ np.diag([math.exp(a), -math.exp(-a)]) @ R.T
    d_theta = dR @ diagonal @ R.T + R @ diagonal @ dR.T
    return (G, d_a, d_theta)

def predict(points, xy, values, parameters, derivative=False, parameter_gradient=False):
    G, da, dt = metric(parameters)
    rho2 = math.exp(2 * parameters[2])
    power = math.exp(parameters[3])
    delta = points[:, None] - xy[None]
    D = np.einsum('ngi,ij,ngj->ng', delta, G, delta) + rho2
    logw = -power * np.log(D)
    w = np.exp(logw - logw.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    mean = w @ values
    residual = values[None] - mean[:, None]
    if parameter_gradient:
        changes = np.stack((np.einsum('ngi,ij,ngj->ng', delta, da, delta), np.einsum('ngi,ij,ngj->ng', delta, dt, delta), np.full_like(D, 2 * rho2)), -1)
        score = np.concatenate((-power * changes / D[..., None], (-power * np.log(D))[..., None]), -1)
        return (mean, np.einsum('ng,ng,ngp->np', w, residual, score))
    if not derivative:
        return mean
    score = -2 * power * np.einsum('ij,ngj->ngi', G, delta) / D[..., None]
    return (mean, np.einsum('ng,ng,ngi->ni', w, residual, score))

class AnisotropicSourceField(SpatialRadioField):

    def __init__(self, base, path):
        self.base = base
        with np.load(path, allow_pickle=False) as z:
            self.parameters = z['parameters'].copy()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def source_ap(self, points, ap_index, derivative=False):
        mean = np.zeros(len(points))
        gradient = np.zeros((len(points), 2))
        ap_index = np.asarray(ap_index, int)
        for ap in range(4):
            ix = np.where(ap_index == ap)[0]
            if not len(ix):
                continue
            valid = self.base.mask[:, ap] > 0.5
            xy = self.base.xy[valid]
            values = (self.base.values[:, :, 0] + self.base.values[:, :, 1])[valid, ap]
            out = predict(points[ix], xy, values, self.parameters[ap], derivative)
            if derivative:
                mean[ix], gradient[ix] = out
            else:
                mean[ix] = out
        return (mean, gradient) if derivative else mean

def fit(a):
    from trajectory_joint_calibration import field
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    base = field(a.base)
    reports = []
    parameters = []
    initial = np.array([0.0, 0.0, math.log(0.15), 0.0])
    for ap in range(4):
        valid = base.mask[:, ap] > 0.5
        xy = base.xy[valid]
        values = (base.values[:, :, 0] + base.values[:, :, 1])[valid, ap]
        region = np.floor(xy / 2.0).astype(int)
        folds = []
        for cell in sorted(set(map(tuple, region))):
            held = np.all(region == cell, axis=-1)
            train = ~held
            if train.sum() < 8:
                continue
            d = np.linalg.norm(xy[held, None] - xy[None, held], axis=-1)
            n = held.sum()
            i, j = np.where((d > 0.2) & (d < 0.95) & (np.arange(n)[:, None] < np.arange(n)[None]))
            folds.append((xy[train], values[train], xy[held], values[held], i, j))

        def objective(p, derivative=True, regularize=True):
            error = []
            contrast = []
            gradient = []
            cg = []
            for X, Y, test, y, i, j in folds:
                prediction, dp = predict(test, X, Y, p, parameter_gradient=True)
                e = prediction - y
                error.extend(e.tolist())
                gradient.extend(dp.tolist())
                contrast.extend((e[i] - e[j]).tolist())
                cg.extend((dp[i] - dp[j]).tolist())
            error = np.asarray(error)
            contrast = np.asarray(contrast)
            gradient = np.asarray(gradient)
            cg = np.asarray(cg)
            value = float(np.mean(error ** 2) + 0.25 * np.mean(contrast ** 2))
            g = 2 * (error[:, None] * gradient).mean(0) + 0.5 * (contrast[:, None] * cg).mean(0)
            if regularize:
                strength = np.array([0.03, 0.0, 0.01, 0.05])
                d = p - initial
                value += float((strength * d * d).sum())
                g += 2 * strength * d
            return (value, g) if derivative else value
        if ap == 0:
            direction = np.array([0.1, 0.2, 0.3, 0.4])
            direction /= np.linalg.norm(direction)
            p = initial + [0.2, 0.3, 0.0, 0.1]
            h = 1e-05
            _, g = objective(p)
            numeric = (objective(p + h * direction, False) - objective(p - h * direction, False)) / (2 * h)
            difference = float(abs(numeric - g @ direction))
            if difference > 1e-05:
                raise ValueError('OPPO spatial-metric parameter gradient differs')
        candidates = []
        for angle in [0.0, math.pi / 4]:
            start = initial + np.array([0.1, angle, 0.0, 0.0])
            result = minimize(objective, start, jac=True, method='L-BFGS-B', bounds=[(-math.log(4), math.log(4)), (-math.pi, math.pi), (math.log(0.03), math.log(1.0)), (math.log(0.5), math.log(3.0))], options=dict(maxiter=160, ftol=1e-10, gtol=1e-05))
            candidates.append(result)
        best = min(candidates, key=lambda r: r.fun)
        if objective(best.x, False, False) >= objective(initial, False, False):
            selected = initial.copy()
        else:
            selected = best.x.copy()
        parameters.append(selected)
        measured = predict(xy, xy, values, selected)
        G, _, _ = metric(selected)
        reports.append(dict(AP=ap, spatial_metric=G.tolist(), radius_m=math.exp(selected[2]), power=math.exp(selected[3]), source_region_objective=objective(selected, False, False), isotropic_region_objective=objective(initial, False, False), full_RP_MAE_db=float(abs(measured - values).mean()), iterations=int(best.nit), converged=bool(best.success)))
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / 'OPPO_spatial_metric.npz'
    np.savez_compressed(path, parameters=np.asarray(parameters))
    adapted = AnisotropicSourceField(base, path)
    points = base.xy[::30] + [0.13, -0.17]
    aps = np.arange(len(points)) % 4
    h = 1e-05
    direction = np.array([0.6, 0.8])
    _, g = adapted.source_ap(points, aps, True)
    numeric = (adapted.source_ap(points + h * direction, aps) - adapted.source_ap(points - h * direction, aps)) / (2 * h)
    position_difference = float(abs(numeric - g @ direction).max())
    if position_difference > 0.0001:
        raise ValueError('Anisotropic field position gradient differs')
    prior.update(source_map_anisotropic_model=str(path.resolve()), source_map_anisotropic_fit='OPPO-only omitted 2m regions; point and local contrast errors', source_map_anisotropic_parameters=16, source_map_anisotropic_target_data_used=False, source_map_anisotropic_reports=reports, source_map_anisotropic_nonlinear_device_descriptor_frozen=True, source_map_anisotropic_parameter_gradient_difference=difference, source_map_anisotropic_position_gradient_difference=position_difference, source_map_anisotropic_fit_seconds=time.monotonic() - began, cache_solver_groups=True, save_solver_state=True)
    write(a.output / 'source_prior.json', prior)
    print(json.dumps(dict(reports=reports, parameter_gradient_difference=difference, position_gradient_difference=position_difference, seconds=time.monotonic() - began)), flush=True)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    fit(p.parse_args())
if __name__ == '__main__':
    main()
