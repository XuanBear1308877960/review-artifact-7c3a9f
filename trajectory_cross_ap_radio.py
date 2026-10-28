import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from engine_utils import write
from trajectory_joint_calibration import field
from trajectory_joint_marginal import initial_prior, mixture_posterior, score
from trajectory_temporal_radio import ObservationModel, TemporalJointPath
from trajectory_raw_hmm import ForceRawHMMPath
LOWER = [(a, b) for a in range(4) for b in range(a)]

class JointObservations(ObservationModel):

    @staticmethod
    def flatten(x):
        return x.swapaxes(0, 1).reshape((-1,) + x.shape[2:])

    def noise_joint(self, parameters, derivative=False):
        p = np.asarray(parameters)
        N = self.N
        L = np.diag(np.exp(p[4:8]))
        for value, (a, b) in zip(p[13:], LOWER):
            L[a, b] = value
        Sigma = L @ L.T
        tau = math.exp(p[8])
        white = np.exp(2 * p[:4])
        raw = np.exp(2 * p[9:13])
        shared = np.zeros((4 * N, 4 * N))
        T = {}
        dT = {}
        for a in range(4):
            ra = self.raw_times[self.bounds[a]:self.bounds[a + 1]]
            sa = slice(a * N, (a + 1) * N)
            for b in range(a + 1):
                rb = self.raw_times[self.bounds[b]:self.bounds[b + 1]]
                sb = slice(b * N, (b + 1) * N)
                dt = abs(ra[:, None] - rb[None])
                K = np.exp(-dt / tau)
                kernel = self.S[a] @ K @ self.S[b].T
                shared[sa, sb] = Sigma[a, b] * kernel
                if a != b:
                    shared[sb, sa] = shared[sa, sb].T
                if derivative:
                    T[a, b] = kernel
                    T[b, a] = kernel.T
                    dkernel = self.S[a] @ (K * dt / tau) @ self.S[b].T
                    dT[a, b] = dkernel
                    dT[b, a] = dkernel.T
            shared[sa, sa] += raw[a] * self.shared[a]
        C = shared + np.diag(np.repeat(white + 0.1 ** 2, N))
        extra = dict(L=L, Sigma=Sigma, T=T, dT=dT, white=white, raw=raw) if derivative else None
        return (C, shared, extra)

    def whiten_joint(self, mean, B, C, mask=None):
        available = self.mask if mask is None else self.mask & mask
        ix = np.where(self.flatten(available))[0]
        if not len(ix):
            return (np.zeros(0), np.zeros((0, 6)), 0.0, ix, np.zeros((0, 0)))
        L = cholesky(C[np.ix_(ix, ix)], lower=True, check_finite=False)
        residual = self.flatten(self.pre['rssi'] - mean)[ix]
        basis = self.flatten(B)[ix]
        z = solve_triangular(L, residual, lower=True, check_finite=False)
        A = solve_triangular(L, basis, lower=True, check_finite=False)
        return (z, A, float(2 * np.log(np.diag(L)).sum()), ix, L)

    def covariance_gradient(self, G, extra):
        N = self.N
        g = np.zeros(19)
        gS = np.zeros((4, 4))
        for a in range(4):
            sa = slice(a * N, (a + 1) * N)
            g[a] = 2 * extra['white'][a] * np.trace(G[sa, sa])
            g[9 + a] = 2 * extra['raw'][a] * float((G[sa, sa] * self.shared[a]).sum())
            for b in range(4):
                sb = slice(b * N, (b + 1) * N)
                block = G[sa, sb]
                gS[a, b] = float((block * extra['T'][a, b]).sum())
                g[8] += extra['Sigma'][a, b] * float((block * extra['dT'][a, b]).sum())
        gL = (gS + gS.T) @ extra['L']
        g[4:8] = np.diag(gL) * np.diag(extra['L'])
        for i, (a, b) in enumerate(LOWER):
            g[13 + i] = gL[a, b]
        return g

def source_rows(a, prior):
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    radio = field(a.base, prior)
    rows = []
    labels = {}
    for ordinal, s in enumerate(segments):
        name = s['device']
        if name not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target labels in cross-AP covariance fitting')
        if name not in labels:
            labels[name] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / name / 'evaluation_labels.json').read_text())
        folder = a.capture / 'source_train' / s['trajectory_id']
        with np.load(folder / 'PRE_forecasts.npz', allow_pickle=False) as z:
            pre = {k: z[k].copy() for k in z.files}
        saved = json.loads((folder / 'predictions.json').read_text())
        available = [p for p in saved if p['observation_id'] in labels[name]]
        t = np.r_[0.0, [(p['timestamp_ns'] - s['start_ns']) / 1000000000.0 for p in available]]
        xy = np.vstack((s['initial_position_xy_m'], [labels[name][p['observation_id']]['gt_xy_m'] for p in available]))
        model = JointObservations(a.queries / 'source_train' / s['trajectory_id'] / 'radio_queries.npz', pre)
        model.mask &= np.asarray([str(oid) in labels[name] for oid in pre['observation_ids']])[:, None]
        raw_xy = np.stack([np.interp(model.raw_times.clip(0, t[-1]), t, xy[:, j]) for j in range(2)], -1)
        mean, B = model.geometry(raw_xy, radio)
        rows.append(dict(model=model, mean=mean, B=B, prior=initial_prior(pre, prior.get('initial_device_covariance_scale', 1.0)), heldout=ordinal % 5 == 0))
    return rows

def fit(a):
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    rows = source_rows(a, prior)

    def objective(p, heldout, derivative=True):
        total = 0.0
        count = 0
        gradient = np.zeros(19)
        for row in rows:
            if row['heldout'] != heldout:
                continue
            model = row['model']
            C, _, extra = model.noise_joint(p, derivative)
            z, A, ld, ix, L = model.whiten_joint(row['mean'], row['B'], C)
            loss, eta, P, residual, _ = mixture_posterior(z, A, row['prior'], ld)
            total += loss
            count += len(z)
            if derivative:
                precision = cho_solve((L, True), np.eye(len(ix)), check_finite=False)
                basis = model.flatten(row['B'])[ix]
                r = model.flatten(model.pre['rssi'] - row['mean'])[ix] - basis @ eta
                score = precision @ r
                W = precision @ basis
                local = 0.5 * (precision - np.outer(score, score) - W @ P @ W.T)
                G = np.zeros_like(C)
                G[np.ix_(ix, ix)] = local
                gradient += model.covariance_gradient(G, extra)
        return (total / count, gradient / count) if derivative else float(total / count)
    initial = np.r_[prior['temporal_RSSI_parameters'], np.zeros(6)]
    row = rows[0]
    model = row['model']
    C, _, _ = model.noise_joint(initial)
    z, A, ld, _, _ = model.whiten_joint(row['mean'], row['B'], C)
    zero = mixture_posterior(z, A, row['prior'], ld)[0]
    blocks = model.blocks(model.noise(initial[:13])[0])
    z, A, ld = model.whiten(row['mean'], row['B'], blocks)
    block = mixture_posterior(z, A, row['prior'], ld)[0]
    if abs(zero - block) > 1e-06:
        raise ValueError('Zero cross-AP correlations changed parent likelihood')
    direction = np.random.RandomState(82407).normal(size=19)
    direction /= np.linalg.norm(direction)
    h = 0.0001
    _, g = objective(initial, False)
    numeric = (objective(initial + h * direction, False, False) - objective(initial - h * direction, False, False)) / (2 * h)
    if abs(numeric - g @ direction) > 3e-05:
        raise ValueError('Cross-AP covariance source gradient mismatch')
    baseline = objective(initial, True, False)
    best = [baseline, initial.copy()]
    trace = []
    iteration = [0]

    def record(x):
        iteration[0] += 1
        if iteration[0] % 5:
            return
        value = objective(x, True, False)
        trace.append(dict(iteration=iteration[0], source_heldout_NLL=value, seconds=time.monotonic() - began))
        if value < best[0]:
            best[:] = [value, np.asarray(x).copy()]
        print(json.dumps(trace[-1]), flush=True)
    bounds = [(math.log(0.2), math.log(12.0))] * 4 + [(math.log(0.02), math.log(15.0))] * 4 + [(math.log(0.1), math.log(30.0))] + [(math.log(0.02), math.log(12.0))] * 4 + [(-12.0, 12.0)] * 6
    result = minimize(objective, initial, args=(False, True), jac=True, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=80, ftol=1e-07), callback=record)
    value = objective(result.x, True, False)
    if value < best[0]:
        best[:] = [value, result.x.copy()]
    p = best[1]
    _, _, extra = rows[0]['model'].noise_joint(p, True)
    Sigma = extra['Sigma']
    std = np.sqrt(np.diag(Sigma))
    correlation = Sigma / std[:, None] / std[None]
    prior.update(RSSI_cross_AP_noise=True, cross_AP_RSSI_parameters=p.tolist(), cross_AP_source_heldout_NLL=best[0], cross_AP_independent_source_heldout_NLL=baseline, cross_AP_selected=bool(best[0] < baseline), cross_AP_source_trace=trace, cross_AP_propagation_covariance_db2=Sigma.tolist(), cross_AP_propagation_correlation=correlation.tolist(), cross_AP_covariance_target_labels_used=False, cross_AP_fit_seconds=time.monotonic() - began, cross_AP_source_gradient_difference=float(abs(numeric - g @ direction)), cross_AP_zero_correlation_parent_difference=float(abs(zero - block)))
    write(a.output, prior)
    print(json.dumps(dict(selected=prior['cross_AP_selected'], heldout_NLL=best[0], parent_heldout_NLL=baseline, propagation_std_db=std.tolist(), propagation_correlation=correlation.tolist(), seconds=time.monotonic() - began)), flush=True)

class CrossAPJointPath(TemporalJointPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observations = JointObservations(Path(self.prior['temporal_RSSI_query_root']) / args[0].parent.name / args[0].name / 'radio_queries.npz', self.pre)
        self.joint_C, self.joint_shared, _ = self.observations.noise_joint(self.prior['cross_AP_RSSI_parameters'])
        self.joint_ix = np.where(self.observations.flatten(self.observations.mask))[0]
        self.joint_L = cholesky(self.joint_C[np.ix_(self.joint_ix, self.joint_ix)], lower=True, check_finite=False)

    def whiten_current(self, mean, B, ix=None, L=None):
        ix = self.joint_ix if ix is None else ix
        L = self.joint_L if L is None else L
        model = self.observations
        if not len(ix):
            return (np.zeros(0), np.zeros((0, 6)), 0.0)
        z = solve_triangular(L, model.flatten(model.pre['rssi'] - mean)[ix], lower=True, check_finite=False)
        A = solve_triangular(L, model.flatten(B)[ix], lower=True, check_finite=False)
        return (z, A, float(2 * np.log(np.diag(L)).sum()))

    def radio(self, q, mode):
        model = self.observations
        mean, B, dm, db = model.geometry(self.raw_spline @ q, self.radio_field, True)
        z, A, ld = self.whiten_current(mean, B)
        loss, eta, P, residual, components = mixture_posterior(z, A, self.device_prior, ld)
        gm = np.zeros(4 * model.N)
        gb = np.zeros((4 * model.N, 6))
        if len(self.joint_ix):
            score = solve_triangular(self.joint_L.T, residual, lower=False, check_finite=False)
            uncertainty = solve_triangular(self.joint_L.T, A @ P, lower=False, check_finite=False)
            gm[self.joint_ix] = -score
            gb[self.joint_ix] = uncertainty - score[:, None] * eta
        gradient = model.pullback(gm.reshape(4, model.N).T, gb.reshape(4, model.N, 6).transpose(1, 0, 2), dm, db, self.raw_spline)
        return (float(loss), gradient, (eta, P, components))

    def solve_axis(self, alternative, modes):
        predictions, diagnostic = ForceRawHMMPath.solve_axis(self, alternative, modes)
        q, _ = self.unpack(self.last_selected_vector[:-1])
        model = self.observations
        raw_xy = self.raw_spline @ q
        mean, B = model.geometry(raw_xy, self.radio_field)
        z, A, ld = self.whiten_current(mean, B)
        _, eta, P, _, components = mixture_posterior(z, A, self.device_prior, ld)
        _, initial_mu, initial_P, _, _ = mixture_posterior(np.zeros(0), np.zeros((0, 6)), self.device_prior)
        N = model.N
        means = np.broadcast_to(initial_mu, (N, 6)).copy()
        covariances = np.broadcast_to(initial_P, (N, 6, 6)).copy()
        delta = np.zeros((N, 4))
        raw_cells = [tuple(cell) for cell in np.floor(raw_xy / 2.0).astype(int)]
        now_cells = np.floor(self.B @ q / 2.0).astype(int)
        groups = {}
        for i in range(5, N):
            regions = {tuple(now_cells[i])}
            for ap in range(4):
                for j in np.where(model.S[ap][i] > 0)[0]:
                    regions.add(raw_cells[model.bounds[ap] + j])
            groups.setdefault(tuple(sorted(regions)), []).append(i)
        for regions, rows in groups.items():
            mask = model.mask.copy()
            region_set = set(regions)
            for ap in range(4):
                bad = np.asarray([raw_cells[j] in region_set for j in range(model.bounds[ap], model.bounds[ap + 1])])
                mask[model.S[ap][:, bad].sum(-1) > 0, ap] = False
            ix = np.where(model.flatten(mask))[0]
            L = cholesky(self.joint_C[np.ix_(ix, ix)], lower=True, check_finite=False)
            z, A, ld = self.whiten_current(mean, B, ix, L)
            _, mu, cov, residual, _ = mixture_posterior(z, A, self.device_prior, ld)
            means[rows] = mu
            covariances[rows] = cov
            if not len(ix):
                continue
            score = solve_triangular(L.T, residual, lower=False, check_finite=False)
            output = (np.arange(4)[:, None] * N + np.asarray(rows)[None]).ravel()
            delta[rows] = (self.joint_shared[np.ix_(output, ix)] @ score).reshape(4, len(rows)).T
        self.calibration_forecasts = dict(device_mean=means, device_covariance=covariances, short_residual=delta, mode=0)
        diagnostic.update(joint_marginal_device_mean=eta.tolist(), joint_marginal_device_covariance=P.tolist(), initial_mixture_posterior_component_weights=components.tolist(), device_prior_PRE_index=5, measured_RSSI_contributor_clock_used=True, shared_raw_RSSI_noise_modeled=True, RSSI_cross_AP_propagation_modeled=True, rolling_initial_raw_sample_overlap_excluded=True, region_cavity_groups=len(groups), RSSI_FORECAST_timing='Offline raw-contributor estimated-region observation-factor omission with cross-AP covariance', known_phone_heading_used=False)
        return (predictions, diagnostic)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['fit', 'score'], required=True)
    for name in ['base', 'capture', 'queries', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--cohorts', nargs='+', default=['original_fixed20', 'AP_movement'])
    a = p.parse_args()
    fit(a) if a.stage == 'fit' else score(a)
if __name__ == '__main__':
    main()
