"""
Keras (torch backend) losses for voxelmorph

If you use this code, please cite one of the voxelmorph papers:
https://github.com/voxelmorph/voxelmorph/blob/master/citations.bib

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

# third party
import numpy as np
from keras import ops
import neurite as ne


def _ensure_tensor(value, reference=None):
    tensor = ops.convert_to_tensor(value)
    if reference is not None:
        tensor = ops.cast(tensor, ops.dtype(reference))
    return tensor


def _flatten_batch(tensor):
    shape = ops.shape(tensor)
    flat_shape = ops.concatenate([shape[:1], ops.convert_to_tensor([-1], dtype=ops.dtype(shape))], axis=0)
    return ops.reshape(tensor, flat_shape)


def _safe_divide(numerator, denominator, eps=1e-8):
    eps_tensor = ops.convert_to_tensor(eps, dtype=ops.dtype(denominator))
    return ops.where(ops.abs(denominator) > eps_tensor, numerator / denominator, ops.zeros_like(numerator))


def _diff_along_dim(tensor, dim):
    slices_front = [slice(None)] * tensor.ndim
    slices_back = [slice(None)] * tensor.ndim
    slices_front[dim] = slice(1, None)
    slices_back[dim] = slice(0, -1)
    return tensor[tuple(slices_front)] - tensor[tuple(slices_back)]


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, eps=1e-5, signed=False):
        self.win = win
        self.eps = eps
        self.signed = signed

    def ncc(self, Ii, Ji):
        Ii = ops.convert_to_tensor(Ii)
        Ji = ops.convert_to_tensor(Ji)
        Ji = ops.cast(Ji, ops.dtype(Ii))

        ndims = Ii.ndim - 2
        if ndims not in (1, 2, 3):
            raise ValueError(f'volumes should be 1 to 3 dimensions. found: {ndims}')

        if self.win is None:
            win = [9] * ndims
        elif isinstance(self.win, (list, tuple)):
            win = list(self.win)
        else:
            win = [int(self.win)] * ndims

        def _conv(volume):
            channels = volume.shape[-1]
            kernel_shape = (*win, channels, 1)
            kernel = ops.ones(kernel_shape, dtype=ops.dtype(volume))
            strides = (1,) * ndims
            conv = ops.nn.conv(volume, kernel, strides=strides, padding='same', data_format='channels_last')
            return conv

        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = _conv(Ii)
        J_sum = _conv(Ji)
        I2_sum = _conv(I2)
        J2_sum = _conv(J2)
        IJ_sum = _conv(IJ)

        win_prod = np.prod(win)
        channels = Ii.shape[-1]
        if channels is None:
            channels = ops.shape(Ii)[-1]
        if isinstance(channels, int):
            channels_tensor = ops.convert_to_tensor(channels, dtype=ops.dtype(Ii))
        else:
            channels_tensor = ops.cast(channels, ops.dtype(Ii))
        win_size = ops.convert_to_tensor(win_prod, dtype=ops.dtype(Ii)) * channels_tensor

        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        eps_tensor = ops.convert_to_tensor(self.eps, dtype=ops.dtype(cross))
        cross = ops.maximum(cross, eps_tensor)
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = ops.maximum(I_var, eps_tensor)
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = ops.maximum(J_var, eps_tensor)

        if self.signed:
            cc = cross / ops.sqrt(I_var * J_var + eps_tensor)
        else:
            cc = (cross / I_var) * (cross / J_var)

        return cc

    def loss(self, y_true, y_pred, reduce='mean'):
        cc = self.ncc(y_true, y_pred)
        flat = _flatten_batch(cc)
        if reduce == 'mean':
            cc = ops.mean(flat, axis=1)
        elif reduce == 'max':
            cc = ops.max(flat, axis=1)
        elif reduce is not None:
            raise ValueError(f'Unknown NCC reduction type: {reduce}')
        return -cc


class MSE:
    """
    Sigma-weighted mean squared error for image reconstruction.
    """

    def __init__(self, image_sigma=1.0):
        self.image_sigma = image_sigma

    def mse(self, y_true, y_pred):
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        y_pred = ops.cast(y_pred, ops.dtype(y_true))
        return ops.square(y_true - y_pred)

    def loss(self, y_true, y_pred, reduce='mean'):
        mse = self.mse(y_true, y_pred)
        if reduce == 'mean':
            mse = ops.mean(mse)
        elif reduce == 'max':
            mse = ops.max(mse)
        elif reduce is not None:
            raise ValueError(f'Unknown MSE reduction type: {reduce}')
        return 1.0 / (self.image_sigma ** 2) * mse


class TukeyBiweight:
    """
    Tukey-Biweight loss.

    The single parameter c represents the threshold above which voxel
    differences are cropped and have no further effect (that is, they are
    treated as outliers and automatically discounted).

    See: DOI: 10.1016/j.neuroimage.2010.07.020
    Reuter, Rosas and Fischl, 2010. Highly accurate inverse consistent registration: 
    a robust approach. NeuroImage, 53(4):1181-96.
    """

    def __init__(self, c=0.5):
        self.csq = c * c  # squared error threshold

    def loss(self, y_true, y_pred):
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        y_pred = ops.cast(y_pred, ops.dtype(y_true))

        error_sq = ops.square(y_true - y_pred)
        threshold = ops.convert_to_tensor(self.csq, dtype=ops.dtype(error_sq))
        mask_below = ops.cast(error_sq <= threshold, ops.dtype(error_sq))
        rho_above = ops.cast(error_sq > threshold, ops.dtype(error_sq)) * threshold / 2.0

        inner = 1 - (error_sq * mask_below) / threshold
        rho_below = (threshold / 2.0) * (1 - ops.power(inner, 3))
        rho = rho_above + rho_below

        return ops.mean(rho)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        y_pred = _ensure_tensor(y_pred)
        y_true = _ensure_tensor(y_true, y_pred)

        ndims = y_pred.ndim - 2
        vol_axes = tuple(range(1, ndims + 1))

        numerator = 2.0 * ops.sum(y_true * y_pred, axis=vol_axes)
        denominator = ops.sum(y_true + y_pred, axis=vol_axes)
        dice = _safe_divide(numerator, denominator)
        dice = ops.mean(dice)
        return -dice


class Grad:
    """
    N-D gradient loss.
    loss_mult can be used to scale the loss value - this is recommended if
    the gradient is computed on a downsampled vector field (where loss_mult
    is equal to the downsample factor).
    """

    def __init__(self, penalty='l1', loss_mult=None, vox_weight=None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        self.vox_weight = vox_weight

    def _diffs(self, y):
        y = _ensure_tensor(y)
        weight = None
        if self.vox_weight is not None:
            if not hasattr(self, '_vox_weight_tensor') or ops.dtype(self._vox_weight_tensor) != ops.dtype(y):
                self._vox_weight_tensor = _ensure_tensor(self.vox_weight, y)
            weight = self._vox_weight_tensor

        diffs = []
        for axis in range(1, y.ndim - 1):
            diff = _diff_along_dim(y, axis)
            if weight is not None:
                slices = [slice(None)] * weight.ndim
                slices[axis] = slice(1, None)
                weight_slice = weight[tuple(slices)]
                diff = weight_slice * diff
            diffs.append(diff)

        return diffs

    def loss(self, _, y_pred):
        """
        returns Tensor of size [bs]
        """
        diffs = self._diffs(y_pred)
        if self.penalty == 'l1':
            diffs = [ops.abs(f) for f in diffs]
        else:
            if self.penalty != 'l2':
                raise ValueError(f"penalty can only be l1 or l2. Got: {self.penalty}")
            diffs = [ops.square(f) for f in diffs]

        flattened = [ops.mean(_flatten_batch(f), axis=1) for f in diffs]
        grad = ops.mean(ops.stack(flattened, axis=0), axis=0)

        if self.loss_mult is not None:
            grad = grad * self.loss_mult

        return grad

    def mean_loss(self, y_true, y_pred):
        """
        returns Tensor of size ()
        """

        return ops.mean(self.loss(y_true, y_pred))


class KL:
    """
    Kullback–Leibler divergence for probabilistic flows.
    """

    def __init__(self, prior_lambda, flow_vol_shape):
        self.prior_lambda = prior_lambda
        self.flow_vol_shape = flow_vol_shape
        self.D = None

    def _degree_matrix(self, vol_shape, device, dtype):
        ndims = len(vol_shape)
        components = []
        for axis in range(ndims):
            deg = np.full(vol_shape, 2.0, dtype=np.float32)
            if vol_shape[axis] > 1:
                slices = [slice(None)] * ndims
                slices[axis] = 0
                deg[tuple(slices)] = 1.0
                slices[axis] = -1
                deg[tuple(slices)] = 1.0
            else:
                deg.fill(1.0)
            components.append(deg)

        stacked = np.stack(components, axis=-1)
        tensor = ops.convert_to_tensor(stacked, dtype=dtype)
        tensor = ops.expand_dims(tensor, axis=0)
        return tensor

    def prec_loss(self, y_pred):
        y_pred = _ensure_tensor(y_pred)
        ndims = y_pred.shape[-1]
        total = ops.convert_to_tensor(0.0, dtype=ops.dtype(y_pred))
        for axis in range(1, y_pred.ndim - 1):
            diff = _diff_along_dim(y_pred, axis)
            total = total + ops.mean(ops.square(diff))

        return 0.5 * total / ndims

    def loss(self, y_true, y_pred):
        """
        KL loss
        y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
        D (number of dimensions) should be 1, 2 or 3

        y_true is only used to get the shape
        """
        y_pred = _ensure_tensor(y_pred)
        ndims = y_pred.shape[-1] // 2
        mean = y_pred[..., :ndims]
        log_sigma = y_pred[..., ndims:]

        if self.D is None or ops.dtype(self.D) != ops.dtype(y_pred):
            self.D = self._degree_matrix(self.flow_vol_shape, None, ops.dtype(y_pred))

        sigma_term = self.prior_lambda * self.D * ops.exp(log_sigma) - log_sigma
        sigma_term = ops.mean(sigma_term)
        prec_term = self.prior_lambda * self.prec_loss(mean)

        return 0.5 * ndims * (sigma_term + prec_term)


class MutualInformation(ne.metrics.MutualInformation):
    """
    Soft Mutual Information approximation for intensity volumes

    More information/citation:
    - Courtney K Guo. 
      Multi-modal image registration with unsupervised deep learning. 
      PhD thesis, Massachusetts Institute of Technology, 2019.
    - M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
      SynthMorph: learning contrast-invariant registration without acquired images
      IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
      https://doi.org/10.1109/TMI.2021.3116879
    """

    def loss(self, y_true, y_pred):
        return -self.volumes(y_true, y_pred)
