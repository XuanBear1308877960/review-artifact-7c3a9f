import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
import torch
from engine_utils import write
from temporal_radio import ORIGINAL_LOAD_ACTOR, PARENT_RELATIVE
from trajectory_joint_calibration import FiniteRadioField, field as selected_radio_field
from trajectory_temporal_radio import ObservationModel

class TorchFiniteAP:

    def __init__(self, net, field):
        self.net = net
        for name in ['xy', 'mask', 'values', 'centres', 'ap', 'band', 'strength']:
            value = getattr(field, name)
            setattr(self, name, None if value is None else torch.tensor(value, device='cuda:0', dtype=torch.float32))

    def __call__(self, points, context, ap, return_descriptor=False):
        from base_spatial_geometry import mlp_with_directions
        delta = points[:, None] - self.xy[None]
        d2 = delta.square().sum(-1) + 0.15 ** 2
        w = self.mask[:, ap].T / d2
        den = w.sum(-1)
        values = self.values[:, ap, :].permute(1, 0, 2)
        component = (w[..., None] * values).sum(1) / den[:, None]
        dw = -2 * delta / d2[..., None] * w[..., None]
        dc = torch.einsum('mgz,mgc->mcz', dw, values - component[:, None]) / den[:, None, None]
        mean = component[:, 0] + component[:, 1]
        dm = dc[:, 0] + dc[:, 1]
        displacement = points - self.centres[ap]
        radius = displacement.square().sum(-1) + 1.0
        visibility = component[:, 3].clamp(0.01, 0.99)
        scalar = torch.stack((0.5 * radius.log10(), component[:, 0] / 50, component[:, 1] / 10, component[:, 2].log1p() / 5, visibility), -1)
        inputs = torch.cat((points / 10, scalar, self.ap[ap], self.band[ap], context), -1)
        ds = torch.stack((displacement / (math.log(10) * radius[:, None]), dc[:, 0] / 50, dc[:, 1] / 10, dc[:, 2] / (1 + component[:, 2, None]) / 5, dc[:, 3] * ((component[:, 3] > 0.01) & (component[:, 3] < 0.99))[:, None]), -2)
        directions = torch.cat((torch.eye(2, device=points.device)[None].expand(len(points), -1, -1) / 10, ds, torch.zeros((len(points), 42, 2), device=points.device)), -2)
        learned, derivative = mlp_with_directions(self.net.field.basis_net, inputs, directions)
        if self.strength is not None:
            scale = 1 - 0.85 * self.strength.abs()
            learned = scale * learned + self.strength * torch.nn.functional.one_hot(ap, 4)
            derivative *= scale[..., None]
        B = torch.cat((torch.ones_like(mean)[:, None], mean[:, None] / 50, learned), -1)
        dB = torch.cat((torch.zeros_like(dm)[:, None], dm[:, None] / 50, derivative), -2)
        result = (mean, B, dm, dB, visibility)
        return result + (inputs, directions) if return_descriptor else result

class TorchWrappedAP:

    def __init__(self, net, field):
        layers = []
        original = field
        while type(original).__name__ != 'FiniteRadioField':
            name = type(original).__name__
            if name == 'ResidualBasisField':
                convert = lambda x: torch.tensor(x, device='cuda:0', dtype=torch.float32)
                layers.append(('residual', dict(W0=convert(original.W0), b0=convert(original.b0), W1=convert(original.W1), b1=convert(original.b1), amplitude=original.amplitude, scale=convert(original.scale))))
                original = original.base
            elif name == 'RPContrastField':
                models = [tuple((torch.tensor(x, device='cuda:0', dtype=torch.float32) if not isinstance(x, float) else x for x in model)) for model in original.models]
                layers.append(('mean', dict(models=models, strength=original.strength)))
                original = original.base
            elif name == 'BandAffineField':
                layers.append(('affine', dict(band_index=torch.tensor(original.band_index, device='cuda:0', dtype=torch.long))))
                original = original.parent
            else:
                raise ValueError('Selected GPU radio field wrapper not implemented: ' + name)
        self.base = TorchFiniteAP(net, original)
        self.layers = list(reversed(layers))

    def __call__(self, points, context, ap):
        mean, B, dm, dB, visibility, inputs, directions = self.base(points, context, ap, True)
        for kind, layer in self.layers:
            if kind == 'mean':
                delta = torch.zeros_like(mean)
                gradient = torch.zeros_like(dm)
                for a, (centres, coefficient, length) in enumerate(layer['models']):
                    take = ap == a
                    offset = points[take, None] - centres[None]
                    den = offset.square().sum(-1) + length ** 2
                    delta[take] = (length ** 2 / den * coefficient).sum(-1) * layer['strength']
                    gradient[take] = (-2 * length ** 2 * offset / den[..., None].square() * coefficient[None, :, None]).sum(1) * layer['strength']
                mean = mean + delta
                dm = dm + gradient
                B = torch.cat((B[:, :1], mean[:, None] / 50, B[:, 2:]), -1)
                dB = torch.cat((dB[:, :1], dm[:, None] / 50, dB[:, 2:]), -2)
            elif kind == 'residual':
                h = inputs @ layer['W0'].T + layer['b0']
                p = h.sigmoid()
                value = (torch.nn.functional.silu(h) @ layer['W1'].T + layer['b1']).tanh()
                factor = layer['amplitude'] * layer['scale']
                J = torch.einsum('nid,oi->nod', directions, layer['W0']) * (p * (1 + h * (1 - p)))[..., None]
                J = torch.einsum('nid,oi->nod', J, layer['W1']) * (factor * (1 - value.square()))[..., None]
                B = torch.cat((B[:, :2], B[:, 2:] + factor * value), -1)
                dB = torch.cat((dB[:, :2], dB[:, 2:] + J), -2)
            elif kind == 'affine':
                band = layer['band_index'][ap]
                new = torch.zeros_like(B)
                dnew = torch.zeros_like(dB)
                for a in range(2):
                    valid = (band == a).to(B.dtype)
                    new[:, 2 * a] = valid
                    new[:, 2 * a + 1] = valid * mean / 50
                    dnew[:, 2 * a + 1] = valid[:, None] * dm / 50
                B, dB = (new, dnew)
        return (mean, B, dm, dB, visibility)

def potential_energy(xy, grid, potential, weight, bandwidth, derivative=False):
    kernel = -0.5 * np.square(grid[None] - xy[:, None]).sum(-1) / bandwidth ** 2
    ln = logsumexp(kernel + potential, axis=-1)
    ld = logsumexp(kernel, axis=-1)
    value = -float((weight * (ln - ld)).sum())
    if not derivative:
        return value
    pn = np.exp(kernel + potential - ln[:, None])
    pd = np.exp(kernel - ld[:, None])
    gradient = (pd - pn) @ grid / bandwidth ** 2 * weight[:, None]
    return (value, gradient)

def export(a):
    from support_label_barrier import deny_label_reads
    from base_evaluate import AVAILABLE_CPUS
    from base_model import stabilize
    began = time.monotonic()
    torch.set_num_threads(1)
    prior = json.loads(a.prior.read_text())
    if any((prior.get(k) for k in ['source_spatial_GP_model', 'field_decoder_override', 'source_map_anisotropic_model', 'receiver_curvature_sigma_db', 'RSSI_history_noise_source_selected', 'geometry_locked_device_basis', 'RSSI_AP_scale_marginalization'])) or prior.get('device_spatial_basis_strength', 1.0) != 1.0:
        raise ValueError('Extend unsupported GP/anisotropic/curvature/locked-basis/noise equations before GPU feature export')
    net, _, _, _, _ = ORIGINAL_LOAD_ACTOR(a.base, a.base / PARENT_RELATIVE)
    if type(net.localization).__name__ != 'SpatialRSSIHead' or net.localization.history_context or net.temporal_contrast:
        raise ValueError('This controlled reuse expects the actual context-free bounded V79 head')
    field = selected_radio_field(a.base, prior)
    net = net.cuda().eval().requires_grad_(False)
    gpu = TorchWrappedAP(net, field)
    feature_capture = {}
    if a.save_features:
        net.localization.ap.register_forward_pre_hook(lambda module, values: feature_capture.update(features=values[0]))
        torch.save(dict(state_dict={k: v.detach().cpu() for k, v in net.localization.state_dict().items()}, variant=net.localization.variant, history_context=False, geometry_head=False, temporal_contrast=0), a.output / 'initial_head.pt')
    if AVAILABLE_CPUS:
        os.sched_setaffinity(0, AVAILABLE_CPUS)
    segments = json.loads((a.capture / 'segments.json').read_text())
    reports = []
    with deny_label_reads() as audit, torch.no_grad():
        for cohort in a.cohorts:
            for s in segments[cohort]:
                folder = a.capture / cohort / s['trajectory_id']
                cal = a.calibrated / cohort / s['trajectory_id']
                with np.load(cal / 'PRE_forecasts.npz', allow_pickle=False) as z:
                    pre = {k: z[k].copy() for k in z.files}
                with np.load(folder / 'trajectory_factors.npz', allow_pickle=False) as z:
                    grid = z['grid_xy'].copy()
                    ns = z['times_ns'].copy()
                saved = json.loads((cal / 'predictions.json').read_text())
                if [r['observation_id'] for r in saved] != pre['observation_ids'].tolist():
                    raise ValueError('Frozen head calibration/path IDs differ')
                t = (ns - s['start_ns']) / 1000000000.0
                path = np.asarray([r['xy_m'] for r in saved])
                N = len(t)
                Q = len(grid)
                mode = pre['weights'].argmax(-1)
                ix = np.arange(N)
                mu = pre['device_mean'][ix, mode]
                P = pre['device_covariance'][ix, mode]
                forecast = pre['short_residual'][ix, mode]
                obs = ObservationModel(a.queries / cohort / s['trajectory_id'] / 'radio_queries.npz', pre)
                cov, _, _ = obs.noise(np.asarray(prior['temporal_RSSI_parameters']))
                noise = np.stack([c.diagonal() for c in cov], -1)
                support_t = np.r_[0.0, t]
                support_xy = np.vstack((s['initial_position_xy_m'], path))
                raw_xy = np.stack([np.interp(obs.raw_times.clip(0, t[-1]), support_t, support_xy[:, j]) for j in range(2)], -1)
                shift = np.zeros((N, 4, 2, 2))
                context = np.zeros((N, 4, 2, 32))
                contributors = np.zeros((N, 4, 2))
                unique = np.zeros((N, 4))
                for ap in range(4):
                    usage = (obs.S[ap] * obs.mask[:, ap, None]).sum(0)
                    for i in range(N):
                        ids = np.where(obs.S[ap][i] > 0)[0]
                        if len(ids) > 2:
                            raise ValueError('Unexpected median contributor count')
                        for j, rid in enumerate(ids):
                            raw = obs.bounds[ap] + rid
                            contributors[i, ap, j] = obs.S[ap][i, rid]
                            shift[i, ap, j] = raw_xy[raw] - path[i]
                            context[i, ap, j] = obs.raw_context()[raw]
                            if obs.mask[i, ap]:
                                unique[i, ap] += obs.S[ap][i, rid] / max(usage[rid], 1e-12)
                weight = np.minimum(1.0, unique.sum(-1) / obs.mask.sum(-1).clip(1))
                potentials = []
                all_features = []
                all_tangent = []
                all_gaussian = []
                proof = None
                for start in range(0, N, 16):
                    end = min(start + 16, N)
                    B = end - start
                    points = grid[None, :, None, None] + shift[start:end, None]
                    points = torch.tensor(points.reshape(-1, 2), device='cuda:0', dtype=torch.float32)
                    ctx = np.broadcast_to(context[start:end, None], (B, Q, 4, 2, 32)).reshape(-1, 32)
                    ap = np.broadcast_to(np.arange(4)[None, None, :, None], (B, Q, 4, 2)).reshape(-1)
                    arrays = gpu(points, torch.tensor(ctx, device='cuda:0', dtype=torch.float32), torch.tensor(ap, device='cuda:0'))
                    if proof is None:
                        take = min(128, len(points))
                        expected = field.ap_query(points[:take].cpu().numpy().astype(float), ctx[:take].astype(float), ap[:take], True)
                        differences = [float(abs(arrays[j][:take].cpu().numpy() - expected[j]).max()) for j in range(4)]
                        if max(differences) > 0.003:
                            raise ValueError('GPU finite-field forward or spatial derivative differs')
                        proof = differences
                    mean, basis, dm, db, visibility = [v.reshape(B, Q, 4, 2, *v.shape[1:]) for v in arrays]
                    w = torch.tensor(contributors[start:end, None], device='cuda:0', dtype=torch.float32)
                    mean = (mean * w).sum(3)
                    basis = (basis * w[..., None]).sum(3)
                    dm = (dm * w[..., None]).sum(3)
                    db = (db * w[..., None, None]).sum(3)
                    visibility = (visibility * w).sum(3)
                    m = torch.tensor(mu[start:end], device='cuda:0', dtype=torch.float32)
                    pc = torch.tensor(P[start:end], device='cuda:0', dtype=torch.float32)
                    prediction = mean + torch.einsum('bqad,bd->bqa', basis, m) + torch.tensor(forecast[start:end, None], device='cuda:0')
                    gradient = dm + torch.einsum('bqadz,bd->bqaz', db, m)
                    device_cov = torch.einsum('bqad,bde,bqce->bqac', basis, pc, basis)
                    covariance = device_cov + torch.diag_embed(torch.tensor(noise[start:end, None], device='cuda:0', dtype=torch.float32))
                    mask = torch.tensor(obs.mask[start:end, None], device='cuda:0').expand(-1, Q, -1)
                    valid = mask.float()
                    covariance = covariance * valid[..., None] * valid[..., None, :] + torch.diag_embed(1 - valid)
                    y = torch.tensor(pre['rssi'][start:end, None], device='cuda:0').expand(-1, Q, -1)
                    chol = torch.linalg.cholesky(stabilize(covariance))
                    if getattr(a, 'save_gaussian', False):
                        whitened = torch.linalg.solve_triangular(chol, ((y - prediction) * valid)[..., None], upper=False).squeeze(-1)
                        gaussian = -0.5 * whitened.square().sum(-1) - chol.diagonal(dim1=-2, dim2=-1).log().sum(-1) - 2 * math.log(2 * math.pi)
                        all_gaussian.append(gaussian.cpu().numpy())
                    value = net.localization(y, mask, prediction, covariance.diagonal(dim1=-2, dim2=-1), dict(mean=mean, gradient=gradient, source_gradient=dm, visibility=visibility), device_cov.diagonal(dim1=-2, dim2=-1), net.field.ap_embedding.weight, observation_covariance=covariance, observation_cholesky=chol)
                    potentials.append(value.cpu().numpy())
                    if a.save_features:
                        all_features.append(feature_capture['features'].cpu().numpy())
                        white = torch.linalg.solve_triangular(chol, ((y - prediction) * valid)[..., None], upper=False).squeeze(-1)
                        design = torch.linalg.solve_triangular(chol, gradient * valid[..., None], upper=False)
                        root = torch.linalg.cholesky(stabilize(design.transpose(-1, -2) @ design, floor=0.0001))
                        U = torch.linalg.solve_triangular(root, design.transpose(-1, -2), upper=False).transpose(-1, -2)
                        projection = (U.transpose(-1, -2) @ white[..., None]).squeeze(-1)
                        gram = U.transpose(-1, -2) @ U
                        gain = torch.tensor([1.5, 0.7], device='cuda:0')
                        difference = gain - 1
                        M = torch.eye(4, device='cuda:0') + U * difference @ U.transpose(-1, -2)
                        direct = 0.5 * (torch.linalg.slogdet(M)[1] - torch.einsum('bqa,bqac,bqc->bq', white, M - torch.eye(4, device='cuda:0'), white))
                        small = torch.eye(2, device='cuda:0') + difference[:, None] * gram
                        analytic = 0.5 * (torch.linalg.slogdet(small)[1] - (projection.square() * difference).sum(-1))
                        if float(abs(direct - analytic).max()) > 0.0005:
                            raise ValueError('Tangent precision ratio differs from the full AP Gaussian')
                        all_tangent.append(torch.cat((projection.square(), gram.reshape(B, Q, 4)), -1).cpu().numpy())
                potential = np.concatenate(potentials)
                if not np.isfinite(potential).all() or abs(potential).max() > 4.00001:
                    raise FloatingPointError('Invalid frozen bounded head output')
                value, g = potential_energy(path, grid, potential, weight, 0.35, True)
                rng = np.random.RandomState(82629)
                direction = rng.normal(size=path.shape)
                direction /= np.linalg.norm(direction)
                h = 1e-05
                num = (potential_energy(path + h * direction, grid, potential, weight, 0.35) - potential_energy(path - h * direction, grid, potential, weight, 0.35)) / (2 * h)
                difference = float(abs(num - (g * direction).sum()))
                constant = float(abs(potential_energy(path, grid, np.full_like(potential, 0.7), weight, 0.35, True)[1]).max())
                if difference > 2e-05 or constant > 1e-10:
                    raise ValueError('M3 conditional potential creates an invalid spatial gradient')
                out = a.output / 'potentials' / cohort / s['trajectory_id']
                out.mkdir(parents=True, exist_ok=True)
                feature_arrays = dict(features=np.concatenate(all_features), mask=obs.mask, tangent=np.concatenate(all_tangent)) if a.save_features else {}
                if getattr(a, 'save_gaussian', False):
                    signature = hashlib.sha256()
                    for frozen in [mu, P, forecast, obs.mask, shift, context, contributors]:
                        signature.update(np.ascontiguousarray(frozen).tobytes())
                    feature_arrays.update(gaussian_potential=np.concatenate(all_gaussian), fixed_device_mean=mu, fixed_device_covariance=P, fixed_short_residual=forecast, fixed_field_motion=pre['motion'], fixed_M2_sha256=np.asarray(signature.hexdigest()))
                np.savez_compressed(out / 'head.npz', observation_ids=pre['observation_ids'], times_ns=ns, grid_xy=grid, potential=potential, weight=weight, **feature_arrays)
                reports.append(dict(cohort=cohort, trajectory_id=s['trajectory_id'], observations=N, grid_points=Q, field_GPU_differences=proof, head_gradient_difference=difference, constant_potential_gradient_max=constant, potential_quantiles=np.quantile(potential, [0, 0.1, 0.5, 0.9, 1]).tolist(), effective_head_observations=float(weight.sum())))
                if len(reports) % 12 == 0:
                    print(json.dumps(dict(exported=len(reports), seconds=time.monotonic() - began)), flush=True)
        assert not audit['blocked_label_attempts']
    prior.update(frozen_M3_spatial_head_root=str((a.output / 'potentials').resolve()), frozen_M3_spatial_head_scale=0.0, frozen_M3_spatial_head_calibration_root=str(a.calibrated.resolve()), frozen_M3_spatial_head_bandwidth_m=0.35, frozen_M3_spatial_head_source_checkpoint=PARENT_RELATIVE, frozen_M3_spatial_head_Gaussian_recounted=False, frozen_M3_spatial_head_clock='True median contributors; frozen sensor-parent local displacements', frozen_M3_spatial_head_scope='One-pass offline bounded conditional spatial correction; M2 cavity moments, propagation forecasts and local displacements frozen', frozen_M3_spatial_head_target_labels_used=False, cache_solver_groups=True, save_solver_state=True)
    write(a.output / 'base_prior.json', prior)
    for strength in [0.25, 0.5, 1.0]:
        write(a.output / ('head_%03d_prior.json' % round(100 * strength)), dict(prior, frozen_M3_spatial_head_scale=strength))
    write(a.output / 'export_complete.json', dict(status='COMPLETE_LABEL_GUARDED_FROZEN_M3_POTENTIALS', reports=reports, trajectories=len(reports), target_labels_used=False, network_weights_trained=False, saved_head_features=a.save_features, GPU_name=torch.cuda.get_device_name(), GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(), seconds=time.monotonic() - began))
    print(json.dumps(dict(trajectories=len(reports), seconds=time.monotonic() - began)), flush=True)

def select(a):
    segments = json.loads((a.capture / 'segments.json').read_text())['source_train']
    held = [s for i, s in enumerate(segments) if i % 5 == 0]
    labels = {}
    scores = []
    for root in [a.calibrated] + [a.output / ('held_head_%03d' % round(100 * s)) for s in a.strengths]:
        if not (root / 'prediction_complete.json').exists():
            raise ValueError('Source M3 solve incomplete')
        errors = []
        for s in held:
            phone = s['device']
            if phone not in {'oppo2', 'oppo3', 'maprealme', 'vivo'}:
                raise ValueError('Target labels in source head selection')
            if phone not in labels:
                labels[phone] = json.loads((a.base / 'data/processed_rssi_causal_short_v1' / phone / 'evaluation_labels.json').read_text())
            saved = json.loads((root / 'source_train' / s['trajectory_id'] / 'predictions.json').read_text())
            errors.extend((float(np.linalg.norm(np.asarray(p['xy_m']) - labels[phone][p['observation_id']]['gt_xy_m'])) for i, p in enumerate(saved) if i >= 4 and p['observation_id'] in labels[phone]))
        if len(errors) != 654:
            raise ValueError('Source-held scoring cohort changed')
        e = np.asarray(errors)
        metric = dict(n=len(e), median_m=float(np.median(e)), mean_m=float(np.mean(e)), p90_m=float(np.percentile(e, 90)), p95_m=float(np.percentile(e, 95)))
        score = metric['median_m'] + 0.25 * metric['mean_m'] + 0.1 * metric['p90_m'] + 0.1 * metric['p95_m']
        scores.append(dict(root=str(root), metrics=metric, score=score))
    best = min(range(len(scores)), key=lambda i: scores[i]['score'])
    strength = 0.0 if best == 0 else a.strengths[best - 1]
    prior = json.loads((a.output / 'base_prior.json').read_text())
    prior['frozen_M3_spatial_head_scale'] = strength
    selection = dict(scope='Actual source-held 11 full trajectories/654 points', strength=strength, results=scores, target_positions_used=False)
    prior['frozen_M3_spatial_head_source_selection'] = selection
    write(a.output / 'selected_prior.json', prior)
    write(a.output / 'source_selection.json', selection)
    print(json.dumps(dict(selected_strength=strength, metrics=scores[best]['metrics'], baseline=scores[0]['metrics'])), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['export', 'select'], required=True)
    for name in ['base', 'capture', 'queries', 'calibrated', 'prior', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--cohorts', nargs='+', default=['source_train', 'original_fixed20', 'AP_movement'])
    p.add_argument('--lock', type=Path)
    p.add_argument('--strengths', type=float, nargs='+', default=[0.25, 0.5, 1.0])
    p.add_argument('--save-features', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.stage == 'export':
        with a.lock.open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            export(a)
    else:
        select(a)
if __name__ == '__main__':
    main()
