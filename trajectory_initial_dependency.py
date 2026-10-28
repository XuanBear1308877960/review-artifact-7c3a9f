import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp
from engine_utils import write
from trajectory_joint_marginal import initial_prior, posterior

class InitialDependency:

    def __init__(self, folder, pre, configuration, strength=None):
        beta = float(configuration.get('initial_position_device_dependency_strength', 0.0) if strength is None else strength)
        if not 0 <= beta <= 1:
            raise ValueError('Conditional coupling must be between zero and one')
        base = initial_prior(pre, configuration.get('initial_device_covariance_scale', 1.0))
        with np.load(folder / 'native_reference_full25.npz', allow_pickle=False) as z:
            if not np.array_equal(z['observation_ids'], pre['observation_ids']):
                raise ValueError('Native joint and cavity observation IDs differ')
            native_mean = z['means'][5].astype(float)
            native = z['covariances'][5].astype(float)
        if native_mean.shape != (8, 15):
            raise ValueError('Expected the frozen native eight-component 15D state')
        native = 0.5 * (native + native.swapaxes(-1, -2)) + np.eye(15)[None] * 0.0001
        oldL = np.linalg.cholesky(native[:, 9:15, 9:15])
        newL = np.linalg.cholesky(base['covariance'])
        transfer = np.linalg.solve(oldL.swapaxes(-1, -2), newL.swapaxes(-1, -2)).swapaxes(-1, -2)
        cross = transfer @ native[:, 9:15, :2]
        X = native[:, :2, :2]
        T = np.linalg.solve(X, cross.swapaxes(-1, -2)).swapaxes(-1, -2)
        covariance = base['covariance'] - beta ** 2 * T @ cross.swapaxes(-1, -2)
        covariance = 0.5 * (covariance + covariance.swapaxes(-1, -2))
        np.linalg.cholesky(covariance)
        adopted = np.concatenate((np.concatenate((X, cross.swapaxes(-1, -2)), axis=-1), np.concatenate((cross, base['covariance']), axis=-1)), axis=-2)
        np.linalg.cholesky(adopted)
        transported = transfer @ native[:, 9:15, 9:15] @ transfer.swapaxes(-1, -2)
        self.marginal_covariance_difference = float(abs(transported - base['covariance']).max())
        if self.marginal_covariance_difference > 1e-08:
            raise ValueError('Native-to-cavity marginal transport failed')
        self.base = base
        self.native_position_mean = native_mean[:, :2]
        self.T = beta * T
        self.covariance = covariance
        self.precision = np.linalg.inv(covariance)
        self.strength = beta
        self.logdet = np.linalg.slogdet(covariance)[1]
        self.original_device_mean_difference = float(abs(native_mean[:, 9:15] - base['mean']).max())

    def at(self, xy):
        if self.strength == 0:
            return self.base
        mean = self.base['mean'] + np.einsum('kdz,kz->kd', self.T, np.asarray(xy)[None] - self.native_position_mean)
        return dict(mean=mean, covariance=self.covariance, precision=self.precision, vector=np.einsum('kde,ke->kd', self.precision, mean), logdet=self.logdet, log_weight=self.base['log_weight'])

    def posterior(self, z, A, p, logdet, derivative=False):
        values = [posterior(z, A, p, k, logdet) for k in range(8)]
        evidence = p['log_weight'] - np.asarray([v[0] for v in values])
        normalizer = logsumexp(evidence)
        weight = np.exp(evidence - normalizer)
        means = np.asarray([v[1] for v in values])
        covs = np.asarray([v[2] for v in values])
        mean = weight @ means
        delta = means - mean
        covariance = np.einsum('k,kde->de', weight, covs + delta[:, :, None] * delta[:, None, :])
        result = (-float(normalizer), mean, covariance, z - A @ mean, weight)
        if not derivative:
            return result
        grad_mu = np.einsum('kde,ke->kd', p['precision'], p['mean'] - means)
        gradient = np.einsum('k,kdz,kd->z', weight, self.T, grad_mu)
        return result + (gradient,)

def prepare(a):
    from support_label_barrier import deny_label_reads
    began = time.monotonic()
    prior = json.loads(a.prior.read_text())
    segments = json.loads((a.capture / 'segments.json').read_text())
    records = []
    with deny_label_reads() as audit:
        for cohort, items in segments.items():
            for s in items:
                folder = a.capture / cohort / s['trajectory_id']
                with np.load(folder / 'PRE_forecasts.npz', allow_pickle=False) as z:
                    pre = {k: z[k].copy() for k in ['observation_ids', 'device_mean', 'device_covariance', 'weights']}
                bridge = InitialDependency(folder, pre, prior, 1.0)
                rng = np.random.RandomState(82627)
                A = rng.normal(size=(32, 6))
                z = rng.normal(size=32)
                xy = bridge.native_position_mean.mean(0) + [0.13, -0.17]
                direction = np.array([0.6, 0.8])
                h = 1e-05
                p = bridge.at(xy)
                result = bridge.posterior(z, A, p, 0.0, True)
                numeric = (bridge.posterior(z, A, bridge.at(xy + h * direction), 0.0)[0] - bridge.posterior(z, A, bridge.at(xy - h * direction), 0.0)[0]) / (2 * h)
                error = float(abs(numeric - result[-1] @ direction))
                if error > 2e-05:
                    raise ValueError('Candidate-conditioned initial-prior gradient mismatch')
                records.append(dict(cohort=cohort, trajectory_id=s['trajectory_id'], transport_covariance_difference=bridge.marginal_covariance_difference, candidate_position_gradient_difference=error, original_device_mean_difference=bridge.original_device_mean_difference))
        assert not audit['blocked_label_attempts']
    prior.update(initial_position_device_dependency_strength=0.0, initial_dependency_initial_eight_inputs_immutable=True, initial_dependency_native_position_marginal_used=False, initial_dependency_recounts_initial_RSSI=False, initial_dependency_mode_weights='Original eight weights; conditional mixture approximation, not a claim of the exact joint conditional', initial_dependency_native_to_cavity='Parent Cholesky device-marginal transport, native physical marginal retained', initial_dependency_time='Sixth PRE timestamp: first five RSSI already arrived, sixth RSSI not consumed', initial_dependency_target_labels_used=False, cache_solver_groups=True, save_solver_state=True)
    a.output.mkdir(parents=True, exist_ok=True)
    write(a.output / 'base_prior.json', prior)
    for beta in [0.25, 0.5, 1.0]:
        write(a.output / ('coupling_%03d_prior.json' % round(beta * 100)), dict(prior, initial_position_device_dependency_strength=beta))
    write(a.output / 'preparation_complete.json', dict(status='COMPLETE_LABEL_GUARDED_INITIAL_JOINT_TRANSPORT', records=records, trajectories=len(records), target_position_labels_used=False, synthetic_probe='Finite-difference algebra check only; no fabricated sensor values used in inference or fitting', seconds=time.monotonic() - began))
    print(json.dumps(dict(trajectories=len(records), max_transport_difference=max((r['transport_covariance_difference'] for r in records)), max_gradient_difference=max((r['candidate_position_gradient_difference'] for r in records)), seconds=time.monotonic() - began)), flush=True)

def select(a):
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    held = [s for i, s in enumerate(segments) if i % 5 == 0]
    labels = {}
    scores = []
    roots = [a.baseline] + [a.output / ('held_coupling_%03d' % round(100 * s)) for s in a.strengths]
    for root in roots:
        if not (root / 'prediction_complete.json').exists():
            raise ValueError('Source-held dependency replay incomplete')
        errors = []
        for s in held:
            phone = s['device']
            if phone not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
                raise ValueError('Target labels entered initial-dependency source selection')
            if phone not in labels:
                labels[phone] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / phone / 'evaluation_labels.json').read_text())
            saved = json.loads((root / 'source_train' / s['trajectory_id'] / 'predictions.json').read_text())
            errors.extend((float(np.linalg.norm(np.asarray(p['xy_m']) - labels[phone][p['observation_id']]['gt_xy_m'])) for i, p in enumerate(saved) if i >= 4 and p['observation_id'] in labels[phone]))
        if len(errors) != 654:
            raise ValueError('Changed source-held scoring cohort')
        e = np.asarray(errors)
        metric = dict(n=len(e), median_m=float(np.median(e)), mean_m=float(np.mean(e)), p90_m=float(np.percentile(e, 90)), p95_m=float(np.percentile(e, 95)))
        score = metric['median_m'] + 0.25 * metric['mean_m'] + 0.1 * metric['p90_m'] + 0.1 * metric['p95_m']
        scores.append(dict(root=str(root), metrics=metric, score=score))
    best = min(range(len(scores)), key=lambda i: scores[i]['score'])
    strength = 0.0 if best == 0 else a.strengths[best - 1]
    prior = json.loads((a.output / 'base_prior.json').read_text())
    prior['initial_position_device_dependency_strength'] = strength
    selection = dict(scope='Actual source-held full path localization; 11 source trajectories/654 points', strength=strength, results=scores, target_positions_used=False)
    prior['initial_dependency_source_selection'] = selection
    write(a.output / 'selected_prior.json', prior)
    write(a.output / 'source_selection.json', selection)
    print(json.dumps(dict(selected_strength=strength, metrics=scores[best]['metrics'], baseline=scores[0]['metrics'])), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['prepare', 'select'], required=True)
    for name in ['base', 'capture', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--strengths', type=float, nargs='+', default=[0.25, 0.5, 1.0])
    a = p.parse_args()
    prepare(a) if a.stage == 'prepare' else select(a)
if __name__ == '__main__':
    main()
