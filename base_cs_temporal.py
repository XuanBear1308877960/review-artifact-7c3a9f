import numpy as np
import torch
from base_model import stabilize

def configure_filter(configuration, parameters, anchors):
    result = dict(configuration)
    if parameters:
        result['cs_temporal_noise'] = {name: dict(tau_s=float(parameters['tau_s']), stationary_sigma_m=float(parameters['stationary_sigma_m'])) for name in anchors}
    return result

def replace_device(mean, covariance, new_mean, new_covariance):
    dimension = mean.shape[-1]
    if dimension == 15:
        old = stabilize(covariance[..., 9:, 9:])
        old_root = torch.linalg.cholesky(old)
        new_root = torch.linalg.cholesky(stabilize(new_covariance))
        transport = torch.linalg.solve_triangular(old_root.transpose(-1, -2), new_root.transpose(-1, -2), upper=True)
        cross = covariance[..., :9, 9:] @ transport
        full = torch.cat((torch.cat((covariance[..., :9, :9], cross), -1), torch.cat((cross.transpose(-1, -2), new_covariance), -1)), -2)
        return (torch.cat((mean[..., :9], new_mean), -1), stabilize(full))
    outside = list(range(9)) + list(range(15, dimension))
    order = torch.tensor(outside + list(range(9, 15)), device=mean.device)
    reverse = order.argsort()
    count = len(outside)
    permuted = covariance.index_select(-2, order).index_select(-1, order)
    old_root = torch.linalg.cholesky(stabilize(permuted[..., count:, count:]))
    new_root = torch.linalg.cholesky(stabilize(new_covariance))
    transport = torch.linalg.solve_triangular(old_root.transpose(-1, -2), new_root.transpose(-1, -2), upper=True)
    cross = permuted[..., :count, count:] @ transport
    full = torch.cat((torch.cat((permuted[..., :count, :count], cross), -1), torch.cat((cross.transpose(-1, -2), new_covariance), -1)), -2)
    updated = torch.cat((mean.index_select(-1, order)[..., :count], new_mean), -1)
    return (updated.index_select(-1, reverse), stabilize(full.index_select(-2, reverse).index_select(-1, reverse)))

def fit_source_variogram(entries):
    from scipy.optimize import least_squares
    times = []
    differences = []
    adjacent_dt = []
    left = []
    right = []
    all_errors = []
    groups = 0
    for entry in entries:
        rows = entry.get('cs_events')
        if rows is None:
            continue
        for anchor in range(4):
            valid = rows['label_mask'] & (rows['anchor_index'] == anchor)
            t = rows['timestamp_ns'][valid].astype(float) / 1000000000.0
            error = rows['range_m'][valid] - rows['target_range'][valid]
            if len(t) < 3:
                continue
            groups += 1
            centred = error - error.mean()
            all_errors.extend(centred.tolist())
            adjacent_dt.extend(np.diff(t).tolist())
            left.extend(centred[:-1].tolist())
            right.extend(centred[1:].tolist())
            i, j = np.triu_indices(len(t), 1)
            dt = t[j] - t[i]
            keep = (dt > 0.0001) & (dt <= 8.0)
            times.extend(dt[keep].tolist())
            differences.extend((0.5 * (error[j[keep]] - error[i[keep]]) ** 2).tolist())
    times = np.asarray(times)
    differences = np.asarray(differences)
    rows = []
    for lo, hi in zip([0, 0.5, 1, 1.5, 2, 3, 4, 6], [0.5, 1, 1.5, 2, 3, 4, 6, 8]):
        valid = (times > lo) & (times <= hi)
        if valid.sum() < 20:
            continue
        rows.append(dict(lo_s=lo, hi_s=hi, n=int(valid.sum()), dt_s=float(times[valid].mean()), semivariance_m2=float(differences[valid].mean())))
    dt = np.asarray([r['dt_s'] for r in rows])
    value = np.asarray([r['semivariance_m2'] for r in rows])
    weights = np.sqrt(np.asarray([r['n'] for r in rows], float))
    weights /= weights.max()

    def residual(p):
        return weights * (p[1] ** 2 * -np.expm1(-dt / p[0]) + p[2] ** 2 - value)
    fit = least_squares(residual, [2.0, 1.3, 0.7], bounds=([0.2, 0.1, 0.1], [12.0, 4.0, 3.0]))
    long_lag_tau, long_lag_colored, long_lag_white = map(float, fit.x)
    adt = np.asarray(adjacent_dt)
    left = np.asarray(left)
    right = np.asarray(right)
    marginal = float(np.var(all_errors))
    adjacent = []
    for lo, hi in [(0, 0.75), (0.75, 1.5), (1.5, 3.0)]:
        valid = (adt > lo) & (adt <= hi)
        if valid.sum() < 20:
            continue
        adjacent.append(dict(lo_s=lo, hi_s=hi, n=int(valid.sum()), dt_s=float(adt[valid].mean()), correlation=float(np.corrcoef(left[valid], right[valid])[0, 1])))
    short_dt = np.asarray([r['dt_s'] for r in adjacent])
    correlation = np.asarray([r['correlation'] for r in adjacent])
    short_weights = np.sqrt(np.asarray([r['n'] for r in adjacent], float))
    short_weights /= short_weights.max()
    short = least_squares(lambda p: short_weights * (p[1] * np.exp(-short_dt / p[0]) - correlation), [2.0, 0.8], bounds=([0.2, 0.05], [8.0, 0.98]))
    tau, fraction = map(float, short.x)
    colored = (marginal * fraction) ** 0.5
    white = (marginal * (1 - fraction)) ** 0.5
    alternatives = []
    for tau_trial in [max(0.25, tau * 0.65), min(8.0, tau * 1.6)]:
        kernel = np.exp(-short_dt / tau_trial)
        part = float(np.clip(np.sum(short_weights ** 2 * kernel * correlation) / np.sum(short_weights ** 2 * kernel ** 2), 0.05, 0.98))
        sigma = (marginal * part) ** 0.5
        white_sigma = (marginal * (1 - part)) ** 0.5
        alternatives.append(dict(tau_s=tau_trial, stationary_sigma_m=sigma, white_sigma_m=white_sigma, white_sigma_scale=white_sigma / (sigma ** 2 + white_sigma ** 2) ** 0.5))
    return dict(source_only=True, target_labels_opened=False, anchor_trajectory_groups=groups, pairs=len(times), bins=rows, adjacent_pairs=len(adt), adjacent_bins=adjacent, within_group_sigma_m=marginal ** 0.5, fit=dict(tau_s=tau, stationary_sigma_m=colored, white_sigma_m=white), fit_method='Source adjacent-error correlation least squares; fixed observed within-group marginal variance', long_lag_semivariogram_fit=dict(tau_s=long_lag_tau, stationary_sigma_m=long_lag_colored, white_sigma_m=long_lag_white, used_for_runtime=False, reason='Long-lag support changes with trajectory length and includes slow drift'), label_caveat='Source labels are noisy; finite-window demeaning can bias short-lag correlation. Descriptive prior, not calibrated confidence intervals.', alternatives=alternatives)
