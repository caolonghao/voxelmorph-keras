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
import torch
import torch.nn.functional as F
import neurite as ne


def _ensure_tensor(value, reference=None):
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if reference is not None:
        tensor = tensor.to(device=reference.device, dtype=reference.dtype)
    return tensor


def _flatten_batch(tensor):
    return tensor.reshape(tensor.shape[0], -1)


def _safe_divide(numerator, denominator, eps=1e-8):
    return torch.where(torch.abs(denominator) > eps, numerator / denominator, torch.zeros_like(numerator))


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
        if not isinstance(Ii, torch.Tensor):
            Ii = torch.as_tensor(Ii)
        if not isinstance(Ji, torch.Tensor):
            Ji = torch.as_tensor(Ji, device=Ii.device, dtype=Ii.dtype)

        ndims = Ii.ndim - 2
        if ndims not in (1, 2, 3):
            raise ValueError(f'volumes should be 1 to 3 dimensions. found: {ndims}')

        if self.win is None:
            win = [9] * ndims
        elif isinstance(self.win, (list, tuple)):
            win = list(self.win)
        else:
            win = [int(self.win)] * ndims

        padding = tuple(w // 2 for w in win)
        conv_fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[ndims]

        def _conv(volume):
            spatial = volume.shape[1:-1]
            channels = volume.shape[-1]
            vol_cf = volume.permute(0, ndims + 1, *range(1, ndims + 1))
            kernel = torch.ones((1, channels, *win), dtype=volume.dtype, device=volume.device)
            conv = conv_fn(vol_cf, kernel, padding=padding)
            return conv.permute(0, *range(2, 2 + ndims), 1)

        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = _conv(Ii)
        J_sum = _conv(Ji)
        I2_sum = _conv(I2)
        J2_sum = _conv(J2)
        IJ_sum = _conv(IJ)

        win_size = float(np.prod(win) * Ii.shape[-1])
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = torch.clamp(cross, min=self.eps)
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = torch.clamp(I_var, min=self.eps)
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = torch.clamp(J_var, min=self.eps)

        if self.signed:
            cc = cross / torch.sqrt(I_var * J_var + self.eps)
        else:
            cc = (cross / I_var) * (cross / J_var)

        return cc

    def loss(self, y_true, y_pred, reduce='mean'):
        cc = self.ncc(y_true, y_pred)
        batch = cc.shape[0]
        flat = cc.reshape(batch, -1)
        if reduce == 'mean':
            cc = flat.mean(dim=1)
        elif reduce == 'max':
            cc = flat.max(dim=1).values
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
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.as_tensor(y_true)
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.as_tensor(y_pred, device=y_true.device, dtype=y_true.dtype)
        return (y_true - y_pred) ** 2

    def loss(self, y_true, y_pred, reduce='mean'):
        mse = self.mse(y_true, y_pred)
        if not isinstance(mse, torch.Tensor):
            mse = torch.as_tensor(mse)
        if reduce == 'mean':
            mse = torch.mean(mse)
        elif reduce == 'max':
            mse = torch.max(mse)
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
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.as_tensor(y_true)
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.as_tensor(y_pred, device=y_true.device, dtype=y_true.dtype)

        error_sq = (y_true - y_pred) ** 2
        threshold = torch.tensor(self.csq, dtype=error_sq.dtype, device=error_sq.device)
        mask_below = (error_sq <= threshold).to(error_sq.dtype)
        rho_above = (error_sq > threshold).to(error_sq.dtype) * threshold / 2.0

        inner = 1 - (error_sq * mask_below) / threshold
        rho_below = (threshold / 2.0) * (1 - inner.pow(3))
        rho = rho_above + rho_below

        return torch.mean(rho)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        y_pred = _ensure_tensor(y_pred)
        y_true = _ensure_tensor(y_true, y_pred)

        ndims = y_pred.ndim - 2
        vol_axes = tuple(range(1, ndims + 1))

        numerator = 2.0 * torch.sum(y_true * y_pred, dim=vol_axes)
        denominator = torch.sum(y_true + y_pred, dim=vol_axes)
        dice = _safe_divide(numerator, denominator)
        dice = torch.mean(dice)
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
            if not isinstance(self.vox_weight, torch.Tensor) or \
               self.vox_weight.device != y.device or \
               self.vox_weight.dtype != y.dtype:
                self.vox_weight = _ensure_tensor(self.vox_weight, y)
            weight = self.vox_weight

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
            diffs = [torch.abs(f) for f in diffs]
        else:
            if self.penalty != 'l2':
                raise ValueError(f"penalty can only be l1 or l2. Got: {self.penalty}")
            diffs = [f * f for f in diffs]

        flattened = [_flatten_batch(f).mean(dim=1) for f in diffs]
        grad = torch.stack(flattened, dim=0).mean(dim=0)

        if self.loss_mult is not None:
            grad = grad * self.loss_mult

        return grad

    def mean_loss(self, y_true, y_pred):
        """
        returns Tensor of size ()
        """

        return torch.mean(self.loss(y_true, y_pred))


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
            deg = torch.full(vol_shape, 2.0, dtype=dtype, device=device)
            if vol_shape[axis] > 1:
                slices = [slice(None)] * ndims
                slices[axis] = 0
                deg[tuple(slices)] = 1.0
                slices[axis] = -1
                deg[tuple(slices)] = 1.0
            else:
                deg.fill_(1.0)
            components.append(deg)

        stacked = torch.stack(components, dim=-1)
        return stacked.unsqueeze(0)

    def prec_loss(self, y_pred):
        y_pred = _ensure_tensor(y_pred)
        ndims = y_pred.shape[-1]

        total = 0.0
        for axis in range(1, y_pred.ndim - 1):
            diff = _diff_along_dim(y_pred, axis)
            total += torch.mean(diff * diff)

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

        if self.D is None or self.D.device != y_pred.device or self.D.dtype != y_pred.dtype:
            self.D = self._degree_matrix(self.flow_vol_shape, y_pred.device, y_pred.dtype)

        sigma_term = self.prior_lambda * self.D * torch.exp(log_sigma) - log_sigma
        sigma_term = torch.mean(sigma_term)
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
