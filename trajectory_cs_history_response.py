import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from engine_utils import write

def load_history(prior, folder, event_ids=None):
    folder = Path(folder)
    root = Path(prior['CS_actual_history_query_root']) / folder.parent.name / folder.name
    with np.load(root / 'CS_history_queries.npz', allow_pickle=False) as z:
        h = {k: z[k].copy() for k in z.files}
    if event_ids is not None and (not np.array_equal(h['observation_ids'], event_ids)):
        raise ValueError('Physical CS history IDs differ')
    if 'CS_history_response_weights' not in prior:
        raise ValueError('This runtime requires the selected sample-order kernel')
    w = np.asarray(prior['CS_history_response_weights'])[None] * h['history_mask']
    total = w.sum(-1, keepdims=True)
    fallback = np.zeros_like(w)
    fallback[:, 0] = 1.0
    h['response_weights'] = np.divide(w, total, out=fallback.copy(), where=total > 1e-12)
    h['noise_weights'] = np.zeros_like(w)
    for i in range(len(w)):
        ix = np.where(h['history_mask'][i])[0]
        order = ix[np.argsort(h['raw_history_m'][i, ix], kind='stable')]
        n = len(order)
        if n % 2:
            h['noise_weights'][i, order[n // 2]] = 1.0
        else:
            h['noise_weights'][i, order[n // 2 - 1:n // 2 + 1]] = 0.5
    h['past_time_moment'] = (np.maximum(-h['history_times_s'], 0) * h['response_weights']).sum(-1)
    return h

def history_subset(history, indices):
    return {key: value[indices] for key, value in history.items()}

def measured_noise_response(t, history):
    query = history['history_times_s']
    noise = history['noise_weights']
    S = np.zeros((len(query), len(t)))
    for i in range(len(query)):
        for j in np.where(noise[i] > 0)[0]:
            found = np.where(abs(t - query[i, j]) <= 1e-07)[0]
            if len(found):
                S[i, found[0]] += noise[i, j]
    return S

def response_basis(distance, mask, derivative=False):
    M, L = distance.shape
    basis = np.zeros((M, 5))
    jac = np.zeros((M, 5, L))
    for i in range(M):
        available = np.where(mask[i])[0]
        if not len(available):
            raise ValueError('Empty physical CS history')
        short = available[:3]
        jac[i, 0, available[0]] = 1.0
        jac[i, 1, short] = 1 / len(short)
        jac[i, 3, available] = 1 / len(available)
        for column, indices in [(2, short), (4, available)]:
            order = indices[np.argsort(distance[i, indices], kind='stable')]
            n = len(order)
            if n % 2:
                jac[i, column, order[n // 2]] = 1.0
            else:
                jac[i, column, order[n // 2 - 1:n // 2 + 1]] = 0.5
        basis[i] = jac[i] @ distance[i]
    return (basis, jac) if derivative else basis

def fit(a):
    prior = json.loads(a.prior.read_text())
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    labels = {}
    rows = []
    old_lags = np.asarray(prior['CS_response_lags_s'])
    counts = []
    for ordinal, s in enumerate(segments):
        name = s['device']
        if name not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
            raise ValueError('Target labels in CS history-response fitting')
        if name not in labels:
            labels[name] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / name / 'evaluation_labels.json').read_text())
        folder = a.capture / 'source_train' / s['trajectory_id']
        saved = json.loads((folder / 'predictions.json').read_text())
        selected = [p for p in saved if p['observation_id'] in labels[name]]
        t = np.r_[0.0, [(p['timestamp_ns'] - s['start_ns']) / 1000000000.0 for p in selected]]
        xy = np.vstack((s['initial_position_xy_m'], [labels[name][p['observation_id']]['gt_xy_m'] for p in selected]))
        with np.load(folder / 'CS_training_contexts.npz', allow_pickle=False) as z:
            cs = {k: z[k].copy() for k in ['observation_ids', 'timestamp_ns', 'range_m', 'anchor_index']}
        with np.load(folder / 'trajectory_factors.npz', allow_pickle=False) as z:
            anchors = z['anchors'].copy()
        with np.load(a.queries / 'source_train' / s['trajectory_id'] / 'CS_history_queries.npz', allow_pickle=False) as z:
            history = {k: z[k].copy() for k in z.files}
        if not np.array_equal(history['observation_ids'], cs['observation_ids']):
            raise ValueError('CS history event IDs differ')
        ct = (cs['timestamp_ns'] - s['start_ns']) / 1000000000.0
        query = history['history_times_s']
        mask = history['history_mask']
        known = ((query >= 0) | ~mask).all(-1) & (ct >= old_lags[-1]) & (ct <= t[-1]) & (mask.sum(-1) == 9)
        past = ct[:, None] - old_lags[None]
        all_points = np.stack([np.interp(query.clip(0, t[-1]), t, xy[:, j]) for j in range(2)], -1)
        old_points = np.stack([np.interp(past.clip(0, t[-1]), t, xy[:, j]) for j in range(2)], -1)
        distance = np.linalg.norm(all_points - anchors[cs['anchor_index']][:, None], axis=-1)
        old_distance = np.linalg.norm(old_points - anchors[cs['anchor_index']][:, None], axis=-1)
        basis = response_basis(distance, mask)
        for anchor in range(4):
            ix = np.where(known & (cs['anchor_index'] == anchor))[0]
            if len(ix) < 4:
                continue
            y = cs['range_m'][ix]
            y = y - y.mean()
            x = [old_distance[ix], distance[ix], basis[ix]]
            rows.append(dict(x=[v - v.mean(0) for v in x], y=y, heldout=ordinal % 5 == 0, trajectory_id=s['trajectory_id'], anchor=anchor))
            counts.append(len(ix))
    if not rows:
        raise ValueError('No completely observed source CS histories')
    held = np.concatenate([np.full(len(r['y']), r['heldout']) for r in rows])
    y = np.concatenate([r['y'] for r in rows])
    fits = []
    names = ['elapsed_4s_kernel', 'actual_sample_order_kernel', 'actual_history_order_statistics']
    for variant, name in enumerate(names):
        x = np.concatenate([r['x'][variant] for r in rows])
        train = ~held

        def objective(w):
            residual = x[train] @ w - y[train]
            return (float(np.sqrt(residual ** 2 + 0.5 ** 2).mean() + 0.01 * (w * w).sum()), x[train].T @ (residual / np.sqrt(residual ** 2 + 0.5 ** 2)) / train.sum() + 0.02 * w)
        initial = np.zeros(x.shape[1])
        initial[0] = 1.0
        result = minimize(objective, initial, jac=True, method='SLSQP', constraints=[dict(type='eq', fun=lambda w: w.sum() - 1, jac=lambda w: np.ones_like(w))], bounds=[(0.0, 1.0)] * len(initial), options=dict(maxiter=200, ftol=1e-10))
        if not result.success:
            raise ValueError('Source CS response did not converge: ' + str(result.message))
        fits.append(dict(model=name, weights=result.x.tolist(), source_train_centered_MAE_m=float(abs(x[train] @ result.x - y[train]).mean()), source_heldout_centered_MAE_m=float(abs(x[held] @ result.x - y[held]).mean()), source_heldout_instant_MAE_m=float(abs(x[held, 0] - y[held]).mean())))
    best = min(fits, key=lambda r: r['source_heldout_centered_MAE_m'])
    accepted = best['model'] != names[0]
    prior.update(CS_actual_history_query_root=str(a.queries.resolve()), CS_actual_history_response_candidate=best['model'], CS_actual_history_response_selected=bool(accepted), CS_actual_history_response_fits=fits, CS_actual_history_source_pairs=len(y), CS_actual_history_source_training_pairs=int((~held).sum()), CS_actual_history_source_heldout_pairs=int(held.sum()), CS_actual_history_source_blocks=len(rows), CS_actual_history_source_target_GT_used=False, CS_actual_history_response_basis_names=['current', 'mean3', 'median3', 'mean9', 'median9'])
    if best['model'] == names[1]:
        prior['CS_history_response_weights'] = best['weights']
    if best['model'] == names[2]:
        prior['CS_history_response_basis_coefficients'] = best['weights']
    write(a.output, prior)
    print(json.dumps(dict(selected=best['model'], accepted=accepted, pairs=len(y), heldout_pairs=int(held.sum()), fits=fits)), flush=True)

def main():
    p = argparse.ArgumentParser()
    for name in ['base', 'capture', 'queries', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    fit(p.parse_args())
if __name__ == '__main__':
    main()
