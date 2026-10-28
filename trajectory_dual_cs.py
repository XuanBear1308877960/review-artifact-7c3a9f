import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import cholesky
from scipy.optimize import minimize
from scipy.special import expit
from engine_utils import write
from trajectory_correlated_cs import covariance, innovation_nll
from trajectory_axial_motion import AxialPath
from trajectory_path import SensorPath

def raw_scale(t, raw):
    result = []
    for now in t:
        values = raw[abs(t - now) <= 1.5]
        if len(values) < 3:
            values = raw[abs(t - now) <= 3.0]
        mad = 1.4826 * np.median(abs(values - np.median(values))) if len(values) else 1.0
        result.append(0.35 + min(float(mad), 10.0))
    return np.asarray(result)

def response_matrix(t, lags, weights):
    S = np.zeros((len(t), len(t)))
    for i, now in enumerate(t):
        for lag, w in zip(lags, weights):
            query = now - lag
            if query < t[0] or query > t[-1] or (not w):
                continue
            right = int(np.searchsorted(t, query))
            if right == 0:
                S[i, 0] += w
            elif right == len(t):
                S[i, -1] += w
            else:
                left = right - 1
                fraction = (query - t[left]) / max(t[right] - t[left], 1e-08)
                S[i, left] += w * (1 - fraction)
                S[i, right] += w * fraction
    return S

def joint_noise(p, b, prior):
    scale = np.asarray(b['raw_sigma_pred'], float) if 'raw_sigma_pred' in b else np.exp(p[b['anchor']]) * b['raw_scale']
    diagonal = scale ** 2 + 0.05 ** 2
    bias_sigma = np.exp(p[4])
    shared = math.tanh(p[5])
    cross = shared * diagonal[:, None] * b['response'].T
    corrected = covariance(np.asarray(prior['CS_temporal_innovation_parameters']), b['dt'], b['sigma'], b['past_mass'], b.get('past_time_moment'))
    corrected = corrected * math.exp(2 * p[12]) + shared ** 2 * (b['response'] * diagonal[None]) @ b['response'].T
    n = len(diagonal)
    K = np.zeros((2 * n, 2 * n))
    K[:n, :n] = np.diag(diagonal) + bias_sigma ** 2
    K[:n, n:] = cross
    K[n:, :n] = cross.T
    K[n:, n:] = corrected
    L = cholesky(K, lower=True, check_finite=False)
    pi = np.r_[np.asarray(b['raw_pi_pred'], float) if 'raw_pi_pred' in b else np.full(n, expit(p[7])), b['pi']]
    return (L, pi)

def metadata(anchor, t, raw, sigma, pi, past_mass, prior, history=None):
    response = response_matrix(t, np.asarray(prior['CS_response_lags_s']), np.asarray(prior['CS_response_weights']))
    if history is not None:
        from trajectory_cs_history_response import measured_noise_response
        response = measured_noise_response(t, history)
    return dict(anchor=anchor, raw_scale=raw_scale(t, raw), response=response, dt=abs(t[:, None] - t[None, :]), sigma=sigma, pi=np.clip(pi, 1e-05, 1 - 1e-05), past_mass=past_mass, past_time_moment=None if history is None else history['past_time_moment'])

def source_blocks(a, prior):
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    labels = {}
    blocks = []
    lags = np.asarray(prior['CS_response_lags_s'])
    weights = np.asarray(prior['CS_response_weights'])
    for ordinal, s in enumerate(segments):
        name = s['device']
        if name not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target label in raw/corrected likelihood fit')
        if name not in labels:
            labels[name] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / name / 'evaluation_labels.json').read_text())
        folder = a.capture / 'source_train' / s['trajectory_id']
        pred = json.loads((folder / 'predictions.json').read_text())
        rows = [r for r in pred if r['observation_id'] in labels[name]]
        t = np.r_[0.0, [(r['timestamp_ns'] - s['start_ns']) / 1000000000.0 for r in rows]]
        xy = np.vstack((s['initial_position_xy_m'], [labels[name][r['observation_id']]['gt_xy_m'] for r in rows]))
        with np.load(folder / 'CS_training_contexts.npz', allow_pickle=False) as z:
            cs = {k: z[k].copy() for k in z.files}
        with np.load(folder / 'trajectory_factors.npz', allow_pickle=False) as z:
            f = {k: z[k].copy() for k in z.files}
        ct = (cs['timestamp_ns'] - s['start_ns']) / 1000000000.0
        query = ct[:, None] - lags[None]
        event_weights = np.broadcast_to(weights, query.shape)
        history = None
        if prior.get('CS_actual_history_query_root'):
            from trajectory_cs_history_response import load_history
            history = load_history(prior, folder, cs['observation_ids'])
            query = history['history_times_s']
            event_weights = history['response_weights']
        points = np.stack([np.interp(query.clip(0, t[-1]), t, xy[:, j]) for j in range(2)], -1)
        current = np.stack([np.interp(ct, t, xy[:, j]) for j in range(2)], -1)
        anchor_xy = f['anchors'][cs['anchor_index']]
        true_raw = np.linalg.norm(current - anchor_xy, axis=-1)
        true_corrected = (np.linalg.norm(points - anchor_xy[:, None], axis=-1) * event_weights).sum(-1)
        for anchor in range(4):
            ix = np.where((cs['anchor_index'] == anchor) & (ct <= t[-1]))[0]
            if not len(ix):
                continue
            raw = cs['features'][ix, 0] * 20
            if history is not None:
                from trajectory_cs_history_response import history_subset
                selected_history = history_subset(history, ix)
            else:
                selected_history = None
            b = metadata(anchor, ct[ix], raw, f['cs_sigma'][ix], f['cs_pi'][ix], ((query[ix] < 0) * event_weights[ix]).sum(-1), prior, selected_history)
            b.update(raw=raw, corrected=cs['range_m'][ix], true_raw=true_raw[ix], true_corrected=true_corrected[ix], heldout=ordinal % 5 == 0, trajectory_id=s['trajectory_id'], event_indices=ix, source_reference_xy=current[ix], surveyed_anchor_xy=f['anchors'][anchor].copy())
            blocks.append(b)
    return blocks

def fit(a):
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    blocks = source_blocks(a, prior)
    Gmean = np.asarray(prior['CS_temporal_innovation_parameters'])[4:]

    def objective(p, heldout):
        total = 0.0
        count = 0
        for b in blocks:
            if b['heldout'] != heldout:
                continue
            L, pi = joint_noise(p, b, prior)
            raw = b['raw'] - math.exp(p[6]) * b['true_raw'] - p[8 + b['anchor']]
            corrected = b['corrected'] - b['true_corrected'] - Gmean[b['anchor']]
            total += innovation_nll(np.r_[raw, corrected], L, pi)
            count += len(raw) * 2
        return total / count
    means = [float(np.median(np.concatenate([b['raw'] - b['true_raw'] for b in blocks if not b['heldout'] and b['anchor'] == j]))) for j in range(4)]
    initial = np.r_[np.log([1.0] * 4 + [1.5]), np.arctanh(0.2), 0.0, math.log(0.15 / 0.85), means, 0.0]
    bounds = [(math.log(0.1), math.log(5.0))] * 4 + [(math.log(0.1), math.log(8.0)), (-2, 2), (math.log(0.5), math.log(2.0)), (-5, 1)] + [(-10, 10)] * 4 + [(math.log(0.3), math.log(2.0))]
    result = minimize(objective, initial, args=(False,), bounds=bounds, method='L-BFGS-B', options=dict(maxiter=100, ftol=3e-07))
    if not np.isfinite(result.fun):
        raise FloatingPointError('Nonfinite dual CS source fit')
    independent = result.x.copy()
    independent[5] = 0.0
    indices = [j for j in range(len(independent)) if j != 5]

    def independent_objective(p, heldout):
        expanded = independent.copy()
        expanded[indices] = p
        return objective(expanded, heldout)
    separate = minimize(independent_objective, independent[indices], args=(False,), bounds=[bounds[j] for j in indices], method='L-BFGS-B', options=dict(maxiter=80, ftol=3e-07))
    independent[indices] = separate.x
    shared_heldout = objective(result.x, True)
    independent_heldout = objective(independent, True)
    selected = result.x if shared_heldout <= independent_heldout else independent
    prior.update(dual_CS_parameters=selected.tolist(), dual_CS_source_train_NLL=objective(selected, False), dual_CS_shared_source_heldout_NLL=shared_heldout, dual_CS_independent_source_heldout_NLL=independent_heldout, dual_CS_selected_shared_noise=bool(shared_heldout <= independent_heldout), dual_CS_raw_and_corrected_values_replaced=False, dual_CS_later_GT_used=False, dual_CS_source_training_devices=['oppo2', 'oppo3', 'maprealme', 'vivo'], dual_CS_source_fit_seconds=time.monotonic() - began)
    write(a.output, prior)
    print(json.dumps(dict(source_shared_heldout_NLL=shared_heldout, source_independent_heldout_NLL=independent_heldout, selected_shared_noise=prior['dual_CS_selected_shared_noise'], raw_noise_multipliers=np.exp(selected[:4]).tolist(), raw_bias_sigma_m=math.exp(selected[4]), shared_response_coefficient=math.tanh(selected[5]), raw_gain=math.exp(selected[6]), raw_outlier_probability=float(expit(selected[7])), raw_anchor_mean_m=selected[8:12].tolist(), corrected_noise_multiplier=math.exp(selected[12]), seconds=time.monotonic() - began)), flush=True)

class DualAxialPath(AxialPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with np.load(args[0] / 'CS_training_contexts.npz', allow_pickle=False) as z:
            self.raw_range = z['features'][:, 0].copy() * 20
        with np.load(args[0] / 'trajectory_factors.npz', allow_pickle=False) as z:
            sigma = z['cs_sigma'].copy()
        self.dual = np.asarray(self.prior['dual_CS_parameters'])
        self.dual_blocks = []
        self.raw_anchor = self.anchor.copy()
        self.raw_height2 = np.zeros(len(self.ct))
        if 'CS_raw_geometry' in self.prior:
            geometry = self.prior['CS_raw_geometry']
            self.raw_anchor += np.asarray(geometry['anchor_shifts_xy_m'])[self.anchor_index]
            self.raw_height2 = np.asarray(geometry['effective_height_squared_m2'])[self.anchor_index]
        quality = None
        if 'raw_quality_output_root' in self.prior:
            folder = Path(self.prior['raw_quality_output_root']) / args[0].parent.name / args[0].name
            with np.load(folder / 'raw_quality.npz', allow_pickle=False) as z:
                quality = {k: z[k].copy() for k in z.files}
            if not np.array_equal(quality['anchor_index'], self.anchor_index):
                raise ValueError('Raw quality anchor IDs differ')
            if not np.allclose((quality['timestamp_ns'] - self.s['start_ns']) / 1000000000.0, self.ct, rtol=0, atol=1e-08):
                raise ValueError('Raw quality event times differ')
        for anchor in range(4):
            ix = np.where(self.anchor_index == anchor)[0]
            if not len(ix):
                continue
            if self.cs_history_metadata is not None:
                from trajectory_cs_history_response import history_subset
                history = history_subset(self.cs_history_metadata, ix)
            else:
                history = None
            b = metadata(anchor, self.ct[ix], self.raw_range[ix], sigma[ix], self.pi[ix], self.cs_past_mass[ix], self.prior, history)
            if quality is not None:
                b['raw_sigma_pred'] = quality['sigma_m'][ix]
                b['raw_pi_pred'] = quality['outlier_probability'][ix]
            L, pi = joint_noise(self.dual, b, self.prior)
            if self.prior.get('raw_only_CS', False):
                L = L[:len(ix), :len(ix)]
                pi = pi[:len(ix)]
            self.dual_blocks.append((anchor, ix, L, pi))

    def path_objective(self, vector, mode):
        value, gradient = SensorPath.objective(self, vector, mode)
        q, _ = self.unpack(vector)
        current = self.C @ q
        delta = current - self.raw_anchor
        d_raw = np.sqrt(np.square(delta).sum(-1) + self.raw_height2 + 1e-08)
        history = np.einsum('mln,nd->mld', self.C_history, q) - self.anchor[:, None]
        history_d = np.sqrt(np.square(history).sum(-1) + 1e-08)
        d_corrected = (history_d * self.cs_response_weights).sum(-1)
        gain = math.exp(self.dual[6])
        Gmean = np.asarray(self.prior['CS_temporal_innovation_parameters'])[4:]
        r_raw = self.raw_range - gain * d_raw - self.dual[8:12][self.anchor_index]
        r_corrected = self.cs_range - d_corrected - Gmean[self.anchor_index]
        raw_derivative = np.zeros(len(self.ct))
        corrected_derivative = np.zeros(len(self.ct))
        for anchor, ix, L, pi in self.dual_blocks:
            residual = r_raw[ix] if self.prior.get('raw_only_CS', False) else np.r_[r_raw[ix], r_corrected[ix]]
            loss, score = innovation_nll(residual, L, pi, True)
            value += loss
            raw_derivative[ix] = -gain * score[:len(ix)]
            if not self.prior.get('raw_only_CS', False):
                corrected_derivative[ix] = -score[len(ix):]
        grad = self.C.T @ (raw_derivative[:, None] * delta / d_raw[:, None])
        grad += np.einsum('mln,mld,ml,m->nd', self.C_history, history / history_d[..., None], self.cs_response_weights, corrected_derivative)
        gradient[:2 * (self.n - 1)] += grad[1:].ravel()
        return (value, gradient)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    fit(a)
if __name__ == '__main__':
    main()
