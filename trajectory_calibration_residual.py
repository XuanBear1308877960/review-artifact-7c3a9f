from pathlib import Path
import numpy as np
from scipy.special import expit

class ResidualBasisField:

    def __init__(self, base, path):
        self.base = base
        self.descriptor = base
        while type(self.descriptor).__name__ != 'FiniteRadioField':
            if not hasattr(self.descriptor, 'base'):
                raise ValueError('Missing frozen device descriptor')
            self.descriptor = self.descriptor.base
        with np.load(Path(path), allow_pickle=False) as z:
            self.W0 = z['weight0'].copy()
            self.b0 = z['bias0'].copy()
            self.W1 = z['weight1'].copy()
            self.b1 = z['bias1'].copy()
            self.amplitude = float(z['amplitude'])
        if self.W0.shape != (32, 49) or self.W1.shape != (4, 32):
            raise ValueError('Unexpected calibration residual shape')
        self.scale = np.ones(4) if self.descriptor.strength is None else 1 - 0.85 * abs(self.descriptor.strength)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def extra(self, points, context, ap_index, derivative=False):
        if derivative:
            from trajectory_calibration_residual_data import descriptors
            inputs, J = descriptors(self.descriptor, points, context, np.asarray(ap_index, int))
        else:
            inputs = self.descriptor.basis_inputs_ap(points, context, ap_index)[1]
        return self.from_descriptor(inputs, J if derivative else None)

    def from_descriptor(self, inputs, J=None):
        h = inputs @ self.W0.T + self.b0
        p = expit(h)
        activation = h * p
        value = np.tanh(activation @ self.W1.T + self.b1)
        output = self.amplitude * self.scale * value
        if J is None:
            return output
        J = np.einsum('mid,oi->mod', J, self.W0)
        J *= (p * (1 + h * (1 - p)))[..., None]
        J = np.einsum('mid,oi->mod', J, self.W1)
        J *= (self.amplitude * self.scale * (1 - value ** 2))[..., None]
        return (output, J)

    def ap_query(self, points, context, ap_index, derivative=False):
        if type(self.base).__name__ == 'FiniteRadioField':
            original = self.base.ap_query(points, context, ap_index, derivative, return_descriptor=True)
            descriptor, directions = original[-2:]
            original = original[:-2]
            if derivative:
                extra, J = self.from_descriptor(descriptor, directions)
            else:
                extra = self.from_descriptor(descriptor)
        else:
            original = self.base.ap_query(points, context, ap_index, derivative)
            if derivative:
                extra, J = self.extra(points, context, ap_index, True)
            else:
                extra = self.extra(points, context, ap_index)
        B = original[1].copy()
        B[:, 2:] += extra
        if not derivative:
            return (original[0], B)
        dB = original[3].copy()
        dB[:, 2:] += J
        return (original[0], B, original[2], dB)

    def __call__(self, points, context, derivative=False):
        original = self.base(points, context, derivative)
        N = len(points)
        B = original[1].copy()
        query = np.repeat(points, 4, axis=0)
        ctx = np.repeat(context, 4, axis=0)
        ap = np.tile(np.arange(4), N)
        if derivative:
            extra, J = self.extra(query, ctx, ap, True)
        else:
            extra = self.extra(query, ctx, ap)
        B[:, :, 2:] += extra.reshape(N, 4, 4)
        if not derivative:
            return (original[0], B)
        dB = original[3].copy()
        dB[:, :, 2:] += J.reshape(N, 4, 4, 2)
        return (original[0], B, original[2], dB)
