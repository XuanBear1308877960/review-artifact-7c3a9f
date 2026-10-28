import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import cholesky, cho_solve
from engine_utils import write

def kernel(x, y, length):
    d = x[:, None] - y[None]
    return length ** 2 / (np.square(d).sum(-1) + length ** 2)

def idw(x, xy, y):
    w = 1 / (np.square(x[:, None] - xy[None]).sum(-1) + 0.15 ** 2)
    return w @ y / w.sum(-1)

class RPContrastField:

    def __init__(self, base, path, strength=1.0):
        self.base = base
        self.strength = float(strength)
        self.models = []
        with np.load(path, allow_pickle=False) as z:
            for ap in range(4):
                self.models.append((z['xy_' + str(ap)].copy(), z['coefficient_' + str(ap)].copy(), float(z['length_' + str(ap)])))

    def __getattr__(self, name):
        return getattr(self.base, name)

    def correction(self, points, ap_index, derivative=False):
        ap_index = np.asarray(ap_index, int)
        value = np.zeros(len(points))
        gradient = np.zeros((len(points), 2))
        for ap, (xy, coefficient, length) in enumerate(self.models):
            ix = np.where(ap_index == ap)[0]
            if not len(ix):
                continue
            delta = points[ix, None] - xy[None]
            den = np.square(delta).sum(-1) + length ** 2
            value[ix] = length ** 2 / den @ coefficient
            if derivative:
                gradient[ix] = (-2 * length ** 2 * delta / den[..., None] ** 2 * coefficient[None, :, None]).sum(1)
        return (self.strength * value, self.strength * gradient) if derivative else self.strength * value

    def source_ap(self, points, ap_index, derivative=False):
        if derivative:
            mean, g = self.base.source_ap(points, ap_index, True)
            d, dg = self.correction(points, ap_index, True)
            return (mean + d, g + dg)
        return self.base.source_ap(points, ap_index) + self.correction(points, ap_index)

    def ap_query(self, points, context, ap_index, derivative=False):
        old = self.base.ap_query(points, context, ap_index, derivative)
        mean = old[0].copy()
        B = old[1].copy()
        if derivative:
            d, dg = self.correction(points, ap_index, True)
        else:
            d = self.correction(points, ap_index)
        mean += d
        B[:, 1] = mean / 50
        if not derivative:
            return (mean, B)
        dm = old[2] + dg
        db = old[3].copy()
        db[:, 1] = dm / 50
        return (mean, B, dm, db)

    def __call__(self, points, context, derivative=False):
        old = self.base(points, context, derivative)
        n = len(points)
        ap = np.tile(np.arange(4), n)
        query = np.repeat(points, 4, axis=0)
        mean = old[0].copy()
        B = old[1].copy()
        if derivative:
            d, dg = self.correction(query, ap, True)
            dg = dg.reshape(n, 4, 2)
        else:
            d = self.correction(query, ap)
        mean += d.reshape(n, 4)
        B[:, :, 1] = mean / 50
        if not derivative:
            return (mean, B)
        dm = old[2] + dg
        db = old[3].copy()
        db[:, :, 1] = dm / 50
        return (mean, B, dm, db)

    def basis_inputs_ap(self, points, context, ap_index):
        mean, inputs = self.base.basis_inputs_ap(points, context, ap_index)
        return (mean + self.correction(points, ap_index), inputs)

def fit(a):
    from trajectory_joint_calibration import field
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    f = field(a.base, prior)
    arrays = {}
    reports = []
    lengths = [0.3, 0.6, 1.2, 2.4]
    for ap in range(4):
        valid = f.mask[:, ap] > 0.5
        xy = f.xy[valid]
        values = (f.values[:, :, 0] + f.values[:, :, 1])[valid, ap]
        regions = np.floor(xy / 2.0).astype(int)
        groups = sorted(set(map(tuple, regions)))
        trials = []
        for length in lengths:
            all_error = []
            all_contrast = []
            baseline = []
            baseline_contrast = []
            for cell in groups:
                held = np.all(regions == cell, axis=-1)
                train = ~held
                if train.sum() < 8 or not held.any():
                    continue
                X, Y = (xy[train], values[train])
                test = xy[held]
                residual = Y - idw(X, X, Y)
                K = kernel(X, X, length) + np.eye(len(X)) * 1e-06
                coefficient = cho_solve((cholesky(K, lower=True), True), residual)
                before = idw(test, X, Y)
                predicted = before + kernel(test, X, length) @ coefficient
                all_error.extend((predicted - values[held]).tolist())
                baseline.extend((before - values[held]).tolist())
                distance = np.linalg.norm(test[:, None] - test[None], axis=-1)
                i, j = np.where((distance > 0.2) & (distance < 0.95) & (np.arange(len(test))[:, None] < np.arange(len(test))[None]))
                all_contrast.extend((predicted[i] - predicted[j] - values[held][i] + values[held][j]).tolist())
                baseline_contrast.extend((before[i] - before[j] - values[held][i] + values[held][j]).tolist())
            error = float(np.sqrt(np.mean(np.square(all_error))))
            contrast = float(np.sqrt(np.mean(np.square(all_contrast))))
            trials.append(dict(length_m=length, source_held_RP_RMSE_db=error, source_held_contrast_RMSE_db=contrast, source_held_RPs=len(all_error), source_held_pairs=len(all_contrast), selection_score=error + 0.5 * contrast, source_base_RMSE_db=float(np.sqrt(np.mean(np.square(baseline)))), source_base_contrast_RMSE_db=float(np.sqrt(np.mean(np.square(baseline_contrast))))))
        chosen = min(trials, key=lambda r: r['selection_score'])
        length = chosen['length_m']
        baseline = f.source_ap(xy, np.full(len(xy), ap))
        residual = values - baseline
        K = kernel(xy, xy, length) + np.eye(len(xy)) * 1e-06
        coefficient = cho_solve((cholesky(K, lower=True), True), residual)
        arrays['xy_' + str(ap)] = xy
        arrays['coefficient_' + str(ap)] = coefficient
        arrays['length_' + str(ap)] = np.asarray(length)
        reconstructed = baseline + kernel(xy, xy, length) @ coefficient
        reports.append(dict(AP=ap, RP_points=len(xy), region_folds=len(groups), trials=trials, selected=chosen, full_RP_reconstruction_MAE_db=float(abs(reconstructed - values).mean()), base_full_RP_MAE_db=float(abs(baseline - values).mean())))
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / 'OPPO_RP_contrast.npz'
    np.savez_compressed(path, **arrays)
    new = RPContrastField(f, path, 1.0)
    points = f.xy[::30] + [0.13, -0.17]
    ap = np.arange(len(points)) % 4
    _, g = new.source_ap(points, ap, True)
    direction = np.array([0.6, 0.8])
    h = 1e-05
    numeric = (new.source_ap(points + h * direction, ap) - new.source_ap(points - h * direction, ap)) / (2 * h)
    difference = float(abs(numeric - g @ direction).max())
    if difference > 0.0001:
        raise ValueError('Source RP contrast derivative differs')
    prior.update(source_RP_contrast_model=str(path.resolve()), source_RP_contrast_strength=1.0, source_RP_contrast_source_only=True, source_RP_contrast_nonlinear_device_descriptor_frozen=True, source_RP_contrast_common_gain_follows_new_mean=True, source_RP_contrast_target_labels_used=False, source_RP_contrast_source_region_rebuild=True, source_RP_contrast_reports=reports, source_RP_contrast_gradient_difference=difference, source_RP_contrast_fit_seconds=time.monotonic() - began, cache_solver_groups=True, save_solver_state=True)
    write(a.output / 'm1_full_prior.json', prior)
    write(a.output / 'm1_half_prior.json', dict(prior, source_RP_contrast_strength=0.5))
    write(a.output / 'source_field_complete.json', dict(status='COMPLETE_OPPO_ONLY_SMOOTH_RP_CONTRAST', reports=reports, gradient_difference=difference, target_measurements_used=False, target_positions_used=False, seconds=time.monotonic() - began))
    print(json.dumps(dict(lengths_m=[r['selected']['length_m'] for r in reports], RP_MAE_db=[r['full_RP_reconstruction_MAE_db'] for r in reports], source_held_RMSE_db=[r['selected']['source_held_RP_RMSE_db'] for r in reports], base_source_held_RMSE_db=[r['selected']['source_base_RMSE_db'] for r in reports], seconds=time.monotonic() - began)), flush=True)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    fit(p.parse_args())
if __name__ == '__main__':
    main()
