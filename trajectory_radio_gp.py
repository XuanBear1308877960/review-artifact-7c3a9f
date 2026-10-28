import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from engine_utils import write

def design(points, centre):
    return np.stack((np.ones(len(points)), 0.5 * np.log10(np.square(points - centre).sum(-1) + 1.0)), -1)

def train_one(x, y, variance, centre, length, noise):
    X = design(x, centre)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    if beta[1] > 0:
        beta = np.array([y.mean(), 0.0])
    residual = y - X @ beta
    amplitude = max(float(np.std(residual)), 1.0)
    d2 = np.square(x[:, None] - x[None]).sum(-1)
    K = amplitude ** 2 * np.exp(-0.5 * d2 / length ** 2) + np.diag(0.25 + noise * variance)
    alpha = cho_solve(cho_factor(K, lower=True, check_finite=False), residual, check_finite=False)
    return dict(beta=beta, alpha=alpha, length=length, amplitude=amplitude)

def predict(points, x, centre, fit, derivative=False):
    d = points[:, None] - x[None]
    K = fit['amplitude'] ** 2 * np.exp(-0.5 * np.square(d).sum(-1) / fit['length'] ** 2)
    mean = design(points, centre) @ fit['beta'] + K @ fit['alpha']
    if not derivative:
        return mean
    radius2 = np.square(points - centre).sum(-1) + 1.0
    dm = fit['beta'][1] * (points - centre) / (math.log(10) * radius2[:, None])
    dm -= np.einsum('ng,ngz,g->nz', K, d, fit['alpha']) / fit['length'] ** 2
    return (mean, dm)

class SpatialRadioField:

    def __init__(self, base, model):
        self.base = base
        with np.load(model, allow_pickle=False) as z:
            self.saved = {k: z[k].copy() for k in z.files}
        self.fits = [dict(beta=self.saved['beta'][a], alpha=self.saved['alpha'][a], length=float(self.saved['length'][a]), amplitude=float(self.saved['amplitude'][a])) for a in range(4)]

    def source_ap(self, points, ap_index, derivative=False):
        ap_index = np.asarray(ap_index, int)
        mu = np.zeros(len(points))
        dm = np.zeros((len(points), 2))
        for ap in range(4):
            ix = np.where(ap_index == ap)[0]
            if not len(ix):
                continue
            out = predict(points[ix], self.saved['xy'], self.saved['centres'][ap], self.fits[ap], derivative)
            if derivative:
                mu[ix], dm[ix] = out
            else:
                mu[ix] = out
        return (mu, dm) if derivative else mu

    def ap_query(self, points, context, ap_index, derivative=False):
        out = self.base.ap_query(points, context, ap_index, derivative)
        B = out[1].copy()
        gp = self.source_ap(points, ap_index, derivative)
        mean, dm = gp if derivative else (gp, None)
        B[:, 1] = mean / 50
        if not derivative:
            return (mean, B)
        db = out[3].copy()
        db[:, 1] = dm / 50
        return (mean, B, dm, db)

    def __call__(self, points, context, derivative=False):
        out = self.base(points, context, derivative)
        B = out[1].copy()
        N = len(points)
        p = np.repeat(points, 4, axis=0)
        ap = np.tile(np.arange(4), N)
        gp = self.source_ap(p, ap, derivative)
        mean, dm = (gp[0].reshape(N, 4), gp[1].reshape(N, 4, 2)) if derivative else (gp.reshape(N, 4), None)
        B[:, :, 1] = mean / 50
        if not derivative:
            return (mean, B)
        db = out[3].copy()
        db[:, :, 1] = dm / 50
        return (mean, B, dm, db)

def fit(a):
    from trajectory_joint_calibration import field
    f = field(a.base)
    x = f.xy
    y = f.values[:, :, 0] + f.values[:, :, 1]
    var = f.values[:, :, 2]
    folds = (np.floor(x[:, 0] / 2).astype(int) + 3 * np.floor(x[:, 1] / 2).astype(int)) % 5
    fits = []
    reports = []
    for ap in range(4):
        candidates = []
        baseline = []
        for fold in range(5):
            train = folds != fold
            test = ~train
            d2 = np.square(x[test, None] - x[None, train]).sum(-1) + 0.15 ** 2
            w = 1 / d2
            prediction = w @ y[train, ap] / w.sum(-1)
            baseline.extend((prediction - y[test, ap]).tolist())
        for length in [0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            for noise in [0.01, 0.1, 0.3, 1.0]:
                error = []
                for fold in range(5):
                    train = folds != fold
                    test = ~train
                    model = train_one(x[train], y[train, ap], var[train, ap], f.centres[ap], length, noise)
                    prediction = predict(x[test], x[train], f.centres[ap], model)
                    error.extend((prediction - y[test, ap]).tolist())
                candidates.append(dict(length=length, noise=noise, spatial_held_RMSE_db=float(np.sqrt(np.mean(np.square(error))))))
        best = min(candidates, key=lambda r: r['spatial_held_RMSE_db'])
        model = train_one(x, y[:, ap], var[:, ap], f.centres[ap], best['length'], best['noise'])
        fits.append(model)
        reports.append(dict(ap=ap, selected=best, IDW_spatial_held_RMSE_db=float(np.sqrt(np.mean(np.square(baseline)))), candidates=candidates))
    a.output.mkdir(parents=True, exist_ok=True)
    model_path = (a.output / 'source_field.npz').resolve()
    np.savez_compressed(model_path, xy=x, centres=f.centres, **{k: np.asarray([m[k] for m in fits]) for k in ['beta', 'alpha', 'length', 'amplitude']})
    prior = json.loads(a.prior.read_text())
    prior.update(source_spatial_GP_model=str(model_path), source_spatial_GP_fit='Existing OPPO map only; 2m spatial blocks assigned to five folds', source_spatial_GP_target_data_used=False)
    write(a.output / 'field_prior.json', prior)
    write(a.output / 'source_fit.json', dict(reports=reports, map_points=len(x), source='OPPO frozen source map', target_labels_used=False))
    print(json.dumps(dict(map_points=len(x), reports=[{k: v for k, v in r.items() if k != 'candidates'} for r in reports])), flush=True)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    fit(p.parse_args())
if __name__ == '__main__':
    main()
