import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from trajectory_temporal_radio import TemporalJointPath
from trajectory_joint_marginal import mixture_posterior

class CrossRegionPath(TemporalJointPath):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model = self.observations
        parent = Path(self.prior['counterfactual_partition_root']) / args[0].parent.name / args[0].name
        rows = json.loads((parent / 'predictions.json').read_text())
        if [r['observation_id'] for r in rows] != [r['observation_id'] for r in self.predictions]:
            raise ValueError('Counterfactual parent IDs differ')
        self.counterfactual_parent_path = np.asarray([r['xy_m'] for r in rows])
        tt = np.r_[0.0, self.t]
        xy = np.vstack((self.initial, self.counterfactual_parent_path))
        raw_xy = np.stack([np.interp(model.raw_times.clip(0, self.t[-1]), tt, xy[:, j]) for j in range(2)], -1)
        cells = np.floor(raw_xy / float(self.prior.get('counterfactual_region_width_m', 2.0))).astype(int)
        regions = sorted(set(map(tuple, cells)))
        held = []
        for region in regions:
            hit = np.all(cells == region, axis=-1)
            mask = np.zeros_like(model.mask)
            for ap in range(4):
                sl = slice(model.bounds[ap], model.bounds[ap + 1])
                mask[:, ap] = model.S[ap][:, hit[sl]].sum(-1) > 0
            mask &= model.mask
            if mask.any():
                held.append((region, mask))
        coverage = sum((mask.astype(float) for _, mask in held)) if held else np.ones_like(model.mask, float)
        self.counterfactual_blocks = []
        for region, mask in held:
            weight = float((1 / coverage[mask]).mean())
            context = model.mask & ~mask
            blocks = model.blocks(self.radio_covariances, context)
            self.counterfactual_blocks.append((region, weight, blocks, int(mask.sum())))
        self.counterfactual_strength = float(self.prior.get('counterfactual_predictive_strength', 1.0))

    def radio(self, q, mode):
        model = self.observations
        mean, B, dm, db = model.geometry(self.raw_spline @ q, self.radio_field, True)

        def evaluate(blocks):
            z, A, ld = model.whiten(mean, B, blocks)
            loss, eta, P, residual, components = mixture_posterior(z, A, self.device_prior, ld)
            gm = np.zeros_like(mean)
            gb = np.zeros_like(B)
            offset = 0
            for ap, ix, L in blocks:
                end = offset + len(ix)
                score = solve_triangular(L.T, residual[offset:end], lower=False, check_finite=False)
                uncertainty = solve_triangular(L.T, A[offset:end] @ P, lower=False, check_finite=False)
                gm[ix, ap] = -score
                gb[ix, ap] = uncertainty - score[:, None] * eta
                offset = end
            return (float(loss), gm, gb, (eta, P, components))
        loss, gm, gb, posterior = evaluate(self.radio_blocks)
        strength = self.counterfactual_strength
        if self.counterfactual_blocks:
            all_loss = loss
            all_gm = gm.copy()
            all_gb = gb.copy()
            weight_sum = sum((row[1] for row in self.counterfactual_blocks))
            loss = (1 - strength + strength * weight_sum) * all_loss
            gm = (1 - strength + strength * weight_sum) * all_gm
            gb = (1 - strength + strength * weight_sum) * all_gb
            for _, weight, blocks, _ in self.counterfactual_blocks:
                context_loss, context_gm, context_gb, _ = evaluate(blocks)
                loss -= strength * weight * context_loss
                gm -= strength * weight * context_gm
                gb -= strength * weight * context_gb
        return (float(loss), model.pullback(gm, gb, dm, db, self.raw_spline), posterior)

    def solve_axis(self, alternative, modes):
        predictions, diagnostic = super().solve_axis(alternative, modes)
        diagnostic.update(counterfactual_position_objective=True, counterfactual_predictive_strength=self.counterfactual_strength, counterfactual_partition='Fixed raw-contributor cells from the sensor-only parent trajectory', counterfactual_regions=[dict(cell=[int(x) for x in region], weight=weight, held_factors=count) for region, weight, _, count in self.counterfactual_blocks], counterfactual_initial_support_retained=True, counterfactual_context_and_path_independently_replayed=False, counterfactual_objective_scope='Composite regional conditional likelihood, conditioned on frozen causal contexts and sensor-derived folds; later position labels never used')
        return (predictions, diagnostic)
