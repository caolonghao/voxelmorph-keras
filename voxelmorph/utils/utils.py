"""
Keras (torch backend) utilities for voxelmorph

If you use this code, please cite one of the voxelmorph papers:
https://github.com/voxelmorph/voxelmorph/blob/master/citations.bib

Contact: adalca [at] csail [dot] mit [dot] edu

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

# internal python imports
import math
import os
import warnings
from typing import Optional, Sequence, Tuple, Union

# third party imports
import numpy as np
import torch
from keras import ops
from keras import backend as K
from keras import layers as KL
from keras import Model
from keras import Input

# local imports
import neurite as ne
from .. import layers


TensorLike = Union[torch.Tensor, np.ndarray]


def _to_tensor(x: TensorLike, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert ``x`` to a torch tensor with optional dtype/device casts."""

    if isinstance(x, torch.Tensor):
        tensor = x
    else:
        tensor = torch.as_tensor(x)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if device is not None and tensor.device != device:
        tensor = tensor.to(device)
    return tensor


def _voxel_mesh(shape: Sequence[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a dense meshgrid covering the voxel coordinates for ``shape``."""

    coords = [torch.arange(dim, device=device, dtype=dtype) for dim in shape]
    mesh = torch.meshgrid(*coords, indexing='ij')
    return torch.stack(mesh, dim=-1)


def setup_device(gpuid=None):
    """Select a CUDA device for PyTorch-backed execution.

    Parameters
    ----------
    gpuid : str | int | None
        Device specifier mirroring the historical TensorFlow interface. ``None`` or ``'-1'``
        forces CPU execution. Comma-separated strings enable multi-GPU visibility.

    Returns
    -------
    tuple[str, int]
        Selected device string (``'cpu'`` or ``'cuda:i'``) and the number of visible devices.
    """

    if gpuid is None or gpuid == '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return 'cpu', 1

    if isinstance(gpuid, int):
        device_ids = [gpuid]
    else:
        device_ids = [int(idx.strip()) for idx in str(gpuid).split(',') if idx.strip()]

    if not device_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return 'cpu', 1

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)

    if torch.cuda.is_available():
        primary = device_ids[0]
        return f'cuda:{primary}', max(1, len(device_ids))

    # Fallback to CPU if CUDA is not available despite the request.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    return 'cpu', 1


def value_at_location(x, single_vol=False, single_pts=False, force_post_absolute_val=True):
    """
    Extracts value at given point.

    TODO: needs documentation
    """

    # vol is batch_size, *vol_shape, nb_feats
    # loc_pts is batch_size, nb_surface_pts, D or D+1
    vol, loc_pts = x
    vol_t = _to_tensor(vol, dtype=torch.float32)
    loc_t = _to_tensor(loc_pts, dtype=vol_t.dtype, device=vol_t.device)

    samples = ne.utils.interpn(vol_t, loc_t)
    if force_post_absolute_val:
        samples = torch.abs(samples)

    return samples


###############################################################################
# deformation utilities
###############################################################################


def transform(vol, loc_shift, interp_method='linear', fill_value=None,
              shift_center=True, shape=None):
    """Apply affine or dense transforms to images in N dimensions."""

    if shape is not None and shift_center:
        raise ValueError('`shape` option incompatible with `shift_center=True`')

    vol_t = _to_tensor(vol, dtype=torch.float32)
    shift_t = _to_tensor(loc_shift, dtype=vol_t.dtype, device=vol_t.device)

    if is_affine_shape(shift_t.shape):
        target_shape: Sequence[int]
        if shape is None:
            target_shape = tuple(int(d) for d in vol_t.shape[:-1])
        else:
            target_shape = tuple(shape)
        shift_t = affine_to_dense_shift(shift_t, target_shape, shift_center=shift_center)

    if shift_t.ndim < 2:
        raise ValueError('loc_shift must have at least two dimensions (spatial, vector)')

    channelwise = shift_t.ndim == vol_t.ndim + 1
    if channelwise:
        spatial_shape = shift_t.shape[:-2]
    else:
        spatial_shape = shift_t.shape[:-1]

    if shift_t.shape[-1] != len(spatial_shape):
        raise ValueError(
            f'Dimension mismatch: {len(spatial_shape)}D volume with transform of size {shift_t.shape[-1]}.'
        )

    mesh = _voxel_mesh(spatial_shape, device=shift_t.device, dtype=shift_t.dtype)
    mesh = mesh.unsqueeze(-2) if channelwise else mesh
    loc = mesh + shift_t

    return ne.utils.interpn(vol_t, loc, interp_method=interp_method, fill_value=fill_value)


def batch_transform(vol, loc_shift, batch_size=None, interp_method='linear', fill_value=None):
    """ apply transform along batch. Compared to _single_transform, reshape inputs to move the 
    batch axis to the feature/channel axis, then essentially apply single transform, and 
    finally reshape back. Need to know/fix batch_size.

    Important: loc_shift is currently implemented only for shape [B, *new_vol_shape, C, D]. 
        to implement loc_shift size [B, *new_vol_shape, D] (as transform() supports), 
        we need to figure out how to deal with the second-last dimension.

    Other notes:
        - We couldn't use ne.utils.flatten_axes() because that computes the axes size from
          dynamic shapes, whereas we receive the batch size as an explicit integer.
        - There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
          indexing. Due to inconsistencies in how some functions and layers handled xy-indexing, we
          removed it in favor of default ij-indexing to minimize the potential for confusion.

    Args:
        vol (Tensor): volume with size vol_shape or [B, *vol_shape, C]
            where C is the number of channels
        loc_shift: shift volume [B, *new_vol_shape, C, D]
            where C is the number of channels, and D is the dimensionality len(vol_shape)
            If loc_shift is [*new_vol_shape, D], it applies to all channels of vol
        interp_method (default:'linear'): 'linear', 'nearest'
        fill_value (default: None): value to use for points outside the domain.
            If None, the nearest neighbors will be used.

    Return:
        new interpolated volumes in the same size as loc_shift[0]

    Keywords:
        interpolation, sampler, resampler, linear, bilinear
    """
    # input management
    vol_t = _to_tensor(vol, dtype=torch.float32)
    shift_t = _to_tensor(loc_shift, dtype=vol_t.dtype, device=vol_t.device)

    if vol_t.ndim < 3:
        raise ValueError('vol must have shape [batch, *spatial, channels]')

    if batch_size is None:
        batch_size = int(vol_t.shape[0])
    elif batch_size != int(vol_t.shape[0]):
        raise ValueError(f'Tensor has wrong batch size {vol_t.shape[0]} instead of {batch_size}')

    outputs = []
    for b in range(batch_size):
        outputs.append(transform(vol_t[b], shift_t[b], interp_method=interp_method, fill_value=fill_value))

    return torch.stack(outputs, dim=0)


def compose(transforms, interp_method='linear', shift_center=True, shape=None):
    """
    Compose a single transform from a series of transforms.

    Supports both dense and affine transforms, and returns a dense transform unless all
    inputs are affine. The list of transforms to compose should be in the order in which
    they would be individually applied to an image. For example, given transforms A, B,
    and C, to compose a single transform T, where T(x) = C(B(A(x))), the appropriate
    function call is:

    T = compose([A, B, C])

    Parameters:
        transforms: List or tuple of affine and/or dense transforms to compose.
        interp_method: Interpolation method. Must be 'linear' or 'nearest'.
        shift_center: Shift grid to image center when converting matrices to dense transforms.
        shape: ND output shape used for converting matrices to dense transforms. Includes only the
            N spatial dimensions. Only used once, if the rightmost transform is a matrix. If None
            or if the rightmost transform is a warp, the shape of the rightmost warp will be used.
            Incompatible with `shift_center=True`.

    Returns:
        Composed affine or dense transform.

    Notes:
        There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
        indexing. Due to inconsistencies in how some functions and layers handled xy-indexing, we
        removed it in favor of default ij-indexing to minimize the potential for confusion.

    """
    if len(transforms) == 0:
        raise ValueError('Compose transform list cannot be empty')

    tensors = [
        _to_tensor(trf, dtype=torch.float32) if not isinstance(trf, torch.Tensor) else trf
        for trf in transforms
    ]

    curr = None
    for nxt in reversed(tensors):
        if curr is None:
            curr = nxt
            continue

        if not is_affine_shape(nxt.shape):
            if is_affine_shape(curr.shape):
                curr = affine_to_dense_shift(
                    curr,
                    shape=nxt.shape[:-1] if shape is None else shape,
                    shift_center=shift_center,
                )
            curr = curr + transform(nxt, curr, interp_method=interp_method)

        elif not is_affine_shape(curr.shape):
            curr = affine_to_dense_shift(
                nxt,
                shape=curr.shape[:-1],
                shift_center=shift_center,
                warp_right=torch.broadcast_to(curr, curr.shape),
            )

        else:
            nxt_sq = make_square_affine(nxt)
            curr_sq = make_square_affine(curr)
            product = torch.matmul(nxt_sq, curr_sq)
            curr = product[..., :-1, :]

    if curr is None:
        raise ValueError('Compose transform list cannot be empty')

    return curr


def rescale_dense_transform(transform, factor, interp_method='linear'):
    """
    Rescales a dense transform. This involves resizing and rescaling the vector field.

    Parameters:
        transform: A dense warp of shape [..., D1, ..., DN, N].
        factor: Scaling factor.
        interp_method: Interpolation method. Must be 'linear' or 'nearest'.
    """

    trf = _to_tensor(transform, dtype=torch.float32)

    def _resize(field: torch.Tensor) -> torch.Tensor:
        resized = ne.utils.resize(field, factor, interp_method=interp_method)
        return resized * factor

    if trf.ndim == trf.shape[-1] + 1:
        return _resize(trf)

    fields = [_resize(trf[b]) for b in range(trf.shape[0])]
    return torch.stack(fields, dim=0)


def integrate_vec(vec, time_dep=False, method='ss', **kwargs):
    """Integrate a vector field using scaling-and-squaring or quadrature."""

    method = method.lower()
    if method not in {'ss', 'scaling_and_squaring', 'quadrature', 'ode'}:
        raise ValueError("Unsupported integration method; choose from {'ss', 'quadrature', 'ode'}")

    field = _to_tensor(vec, dtype=torch.float32)

    if method in {'ss', 'scaling_and_squaring'}:
        nb_steps = int(kwargs.get('nb_steps', 0))
        if nb_steps < 0:
            raise ValueError('nb_steps should be >= 0')

        if time_dep:
            svec = field.movedim(-1, 0)
            if svec.shape[0] != 2 ** nb_steps:
                raise ValueError('2**nb_steps and vector shape do not match for time-dependent SVF')

            svec = svec / (2 ** nb_steps)
            for _ in range(nb_steps):
                even = svec[0::2]
                odd = svec[1::2]
                composed = []
                for base, inc in zip(even, odd):
                    composed.append(transform(inc, base))
                svec = even + torch.stack(composed, dim=0)
            return svec[0]

        disp = field / (2 ** nb_steps) if nb_steps > 0 else field
        for _ in range(nb_steps):
            disp = disp + transform(disp, disp)
        return disp

    if method == 'quadrature':
        nb_steps = int(kwargs.get('nb_steps', 1))
        if nb_steps < 1:
            raise ValueError('nb_steps should be >= 1')

        if time_dep:
            vec_step = field / nb_steps
            disp = vec_step[..., 0]
            for si in range(nb_steps - 1):
                disp = disp + transform(vec_step[..., si + 1], disp)
            return disp

        vec_step = field / nb_steps
        disp = vec_step
        for _ in range(nb_steps - 1):
            disp = disp + transform(vec_step, disp)
        return disp

    raise NotImplementedError('ODE integration is not implemented for the torch backend yet')


def point_spatial_transformer(x, single=False, sdt_vol_resize=1):
    """
    Transforms surface points with a given deformation.
    Note that the displacement field that moves image A to image B will be "in the space of B".
    That is, `trf(p)` tells you "how to move data from A to get to location `p` in B". 
    Therefore, that same displacement field will warp *landmarks* in B to A easily 
    (that is, for any landmark `L(p)`, it can easily find the appropriate `trf(L(p))`
    via interpolation.

    TODO: needs documentation
    """

    # surface_points is a N x D or a N x (D+1) Tensor
    # trf is a *volshape x D Tensor
    surface_points, trf = x
    trf_t = _to_tensor(trf, dtype=torch.float32)
    surface_t = _to_tensor(surface_points, dtype=trf_t.dtype, device=trf_t.device)
    trf_t = trf_t * sdt_vol_resize

    surface_dim = surface_t.shape[-1]
    trf_dim = trf_t.shape[-1]
    if surface_dim not in (trf_dim, trf_dim + 1):
        raise ValueError('Surface point dimensionality incompatible with transform')

    label = None
    if surface_dim == trf_dim + 1:
        label = surface_t[..., -1:]
        surface_t = surface_t[..., :-1]

    diff = ne.utils.interpn(trf_t, surface_t)
    ret = surface_t + diff

    if label is not None:
        ret = torch.cat((ret, label), dim=-1)

    return ret


# TODO: needs work

def keras_transform(img, trf, interp_method='linear', rescale=None):
    """
    Applies a transform to an image. Note that inputs and outputs are
    in tensor format i.e. (batch, *imshape, nchannels).

    # TODO: it seems that the main addition of this function of the SpatialTransformer 
    # or the transform function is integrating it with the rescale operation? 
    # This needs to be incorporated.
    """
    img_input = Input(shape=img.shape[1:])
    trf_input = Input(shape=trf.shape[1:])
    trf_scaled = trf_input if rescale is None else layers.RescaleTransform(rescale)(trf_input)
    y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, trf_scaled])
    return Model([img_input, trf_input], y_img).predict([img, trf])


###############################################################################
# affine utilities
###############################################################################


def is_affine_shape(shape):
    """
    Determine whether the given shape (single-batch) represents an N-dimensional affine matrix of
    shape (M, N + 1), with `N in (2, 3)` and `M in (N, N + 1)`.

    Parameters:
        shape: Tuple or list of integers excluding the batch dimension.
    """
    if len(shape) == 2 and shape[-1] != 1:
        validate_affine_shape(shape)
        return True
    return False


def validate_affine_shape(shape):
    """
    Validate whether the input shape represents a valid affine matrix of shape (..., M, N + 1),
    where N is the number of dimensions, and M is N or N + 1. Throws an error if the shape is
    invalid.

    Parameters:
        shape: Tuple or list of integers.
    """
    ndim = int(shape[-1]) - 1
    rows = int(shape[-2])
    if ndim not in (2, 3):
        raise ValueError(f'Affine matrix must be 2D or 3D, got {ndim}D')
    if rows not in (ndim, ndim + 1):
        raise ValueError(f'{ndim}D affine matrix must have {ndim} or {ndim + 1} rows, got {rows}.')


def make_square_affine(mat):
    """
    Convert an ND affine matrix of shape (..., N, N + 1) to square shape (..., N + 1, N + 1).

    Parameters:
        mat: Affine matrix of shape (..., M, N + 1), where M is N or N + 1.

    Returns:
        out: Affine matrix of shape (..., N + 1, N + 1).
    """
    tensor = _to_tensor(mat, dtype=torch.float32)
    validate_affine_shape(tensor.shape)
    if tensor.shape[-2] == tensor.shape[-1]:
        return tensor

    ndims = tensor.shape[-1] - 1
    batch_shape = tensor.shape[:-2]
    zeros = tensor.new_zeros(*batch_shape, 1, ndims)
    ones = tensor.new_ones(*batch_shape, 1, 1)
    row = torch.cat((zeros, ones), dim=-1)
    return torch.cat((tensor, row), dim=-2)


def affine_add_identity(mat):
    """
    Add the identity matrix to an N-dimensional 'shift' affine.

    Parameters:
        mat: Affine matrix of shape (..., M, N + 1), where M is N or N + 1.

    Returns:
        out: Affine matrix of shape (..., M, N + 1).
    """
    tensor = _to_tensor(mat, dtype=torch.float32)
    rows, ndp1 = tensor.shape[-2:]
    eye = torch.eye(ndp1, dtype=tensor.dtype, device=tensor.device)[:rows]
    return tensor + eye


def affine_remove_identity(mat):
    """
    Subtract the identity matrix from an N-dimensional affine.

    Parameters:
        mat: Affine matrix of shape (..., M, N + 1), where M is N or N + 1.

    Returns:
        out: Affine matrix of shape (..., M, N + 1).
    """
    tensor = _to_tensor(mat, dtype=torch.float32)
    rows, ndp1 = tensor.shape[-2:]
    eye = torch.eye(ndp1, dtype=tensor.dtype, device=tensor.device)[:rows]
    return tensor - eye


def invert_affine(mat):
    """
    Compute the multiplicative inverse of an N-dimensional affine matrix.

    Parameters:
        mat: Affine matrix of shape (..., M, N + 1), where M is N or N + 1.

    Returns:
        out: Affine matrix of shape (..., M, N + 1).
    """
    tensor = _to_tensor(mat, dtype=torch.float32)
    rows = tensor.shape[-2]
    inv = torch.linalg.inv(make_square_affine(tensor))
    return inv[..., :rows, :]


def rescale_affine(mat, factor):
    """
    Rescales affine matrix by some factor.

    Parameters:
        mat: Affine matrix of shape [..., N, N+1].
        factor: Zoom factor.
    """
    tensor = _to_tensor(mat, dtype=torch.float32)
    scaled_translation = tensor[..., -1] * factor
    scaled_translation = torch.unsqueeze(scaled_translation, dim=-1)
    return torch.cat([tensor[..., :-1], scaled_translation], dim=-1)


def affine_to_dense_shift(matrix, shape, shift_center=True, warp_right=None):
    """Convert affine matrices to dense displacement fields."""

    matrix_t = _to_tensor(matrix, dtype=torch.float32)
    target_shape = tuple(int(s) for s in shape)

    ndims = len(target_shape)
    if matrix_t.shape[-1] != ndims + 1:
        raise ValueError(f'Affine ({matrix_t.shape[-1] - 1}D) does not match target shape ({ndims}D).')
    validate_affine_shape(matrix_t.shape)

    device = matrix_t.device
    dtype = matrix_t.dtype

    grid = _voxel_mesh(target_shape, device=device, dtype=dtype)
    if shift_center:
        center = torch.tensor([(dim - 1) / 2.0 for dim in target_shape], device=device, dtype=dtype)
        grid = grid - center

    grid = grid.reshape((1,) * (matrix_t.ndim - 2) + target_shape + (ndims,))
    grid = grid.expand(matrix_t.shape[:-2] + target_shape + (ndims,))

    if warp_right is not None:
        warp = _to_tensor(warp_right, dtype=dtype, device=device)
        grid = grid + warp

    A = matrix_t[..., :ndims, :ndims]
    b = matrix_t[..., :ndims, -1]

    grid_flat = grid.reshape(matrix_t.shape[:-2] + (-1, ndims))
    transformed = torch.matmul(grid_flat, A.transpose(-1, -2)) + b.unsqueeze(-2)
    disp = transformed - grid_flat
    return disp.reshape(matrix_t.shape[:-2] + target_shape + (ndims,))


def angles_to_rotation_matrix(ang, deg=True, ndims=3):
    """
    Construct N-dimensional rotation matrices from angles, where N is 2 or
    3. The direction of rotation for all axes follows the right-hand rule: the
    thumb being the rotation axis, a positive angle defines a rotation in the
    direction pointed to by the other fingers. Rotations are intrinsic, that
    is, applied in the body-centered frame of reference. The function supports
    inputs with or without batch dimensions.

    In 3D, rotations are applied in the order ``R = X @ Y @ Z``, where X, Y,
    and Z are matrices defining rotations about the x, y, and z-axis,
    respectively.

    Arguments:
        ang: Array-like input angles of shape (..., M), specifying rotations
            about the first M axes of space. M must not exceed N. Any missing
            angles will be set to zero. Lists and tuples will be stacked along
            the last dimension.
        deg: Interpret `ang` as angles in degrees instead of radians.
        ndims: Number of spatial dimensions. Must be 2 or 3.

    Returns:
        mat: Rotation matrices of shape (..., N, N) constructed from `ang`.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    if ndims not in (2, 3):
        raise ValueError(f'Affine matrix must be 2D or 3D, but got ndims of {ndims}.')

    if isinstance(ang, (list, tuple)):
        ang = torch.stack([_to_tensor(a, dtype=torch.float32) for a in ang], dim=-1)
    else:
        ang = _to_tensor(ang, dtype=torch.float32)

    if ang.ndim == 0:
        ang = ang.reshape(1)

    num_ang = 1 if ndims == 2 else 3
    if ang.shape[-1] > num_ang:
        raise ValueError(f'Number of angles exceeds value {num_ang} expected for dimensionality.')

    if ang.shape[-1] < num_ang:
        pad_shape = list(ang.shape[:-1]) + [num_ang - ang.shape[-1]]
        padding = ang.new_zeros(pad_shape)
        ang = torch.cat((ang, padding), dim=-1)

    if deg:
        ang = ang * (math.pi / 180.0)

    cos_vals = torch.cos(ang)
    sin_vals = torch.sin(ang)

    if ndims == 2:
        c0 = cos_vals[..., 0:1]
        s0 = sin_vals[..., 0:1]
        row1 = torch.cat((c0, -s0), dim=-1)
        row2 = torch.cat((s0, c0), dim=-1)
        out = torch.stack((row1, row2), dim=-2)
    else:
        c0, c1, c2 = [cos_vals[..., i:i + 1] for i in range(3)]
        s0, s1, s2 = [sin_vals[..., i:i + 1] for i in range(3)]
        one = torch.ones_like(c0)
        zero = torch.zeros_like(c0)

        rot_x = torch.stack((
            torch.cat((one, zero, zero), dim=-1),
            torch.cat((zero, c0, -s0), dim=-1),
            torch.cat((zero, s0, c0), dim=-1),
        ), dim=-2)
        rot_y = torch.stack((
            torch.cat((c1, zero, s1), dim=-1),
            torch.cat((zero, one, zero), dim=-1),
            torch.cat((-s1, zero, c1), dim=-1),
        ), dim=-2)
        rot_z = torch.stack((
            torch.cat((c2, -s2, zero), dim=-1),
            torch.cat((s2, c2, zero), dim=-1),
            torch.cat((zero, zero, one), dim=-1),
        ), dim=-2)
        out = torch.matmul(rot_x, torch.matmul(rot_y, rot_z))

    return out.squeeze(0) if ang.ndim == 1 else out


def params_to_affine_matrix(par,
                            deg=True,
                            shift_scale=False,
                            last_row=False,
                            ndims=3):
    """
    Construct N-dimensional transformation matrices from affine parameters,
    where N is 2 or 3. The transforms operate in a right-handed frame of
    reference, with right-handed intrinsic rotations (see
    angles_to_rotation_matrix for details), and are constructed by matrix
    product ``T @ R @ S @ E``, where T, R, S, and E are matrices representing
    translation, rotation, scale, and shear, respectively. The function
    supports inputs with or without batch dimensions.

    Arguments:
        par: Array-like input parameters of shape (..., M), defining an affine
            transformation in N-D space. The size M of the right-most dimension
            must not exceed ``N * (N + 1)``. This axis defines, in order:
            translations, rotations, scaling, and shearing parameters. In 3D,
            for example, the first three indices specify translations along the
            x, y, and z-axis, and similarly for the remaining parameters. Any
            missing parameters will bet set to identity. Lists and tuples will
            be stacked along the last dimension.
        deg: Interpret input angles as specified in degrees instead of radians.
        shift_scale: Add 1 to any specified scaling parameters. May be
            desirable when the input parameters are estimated by a network.
        last_row: Append the last row and return a full matrix.
        ndims: Number of dimensions. Must be 2 or 3.

    Returns:
        mat: Affine transformation matrices of shape (..., N, N + 1) or
            (..., N + 1, N + 1), depending on `last_row`. The left-most
            dimensions depend on the input shape.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    if ndims not in (2, 3):
        raise ValueError(f'Affine matrix must be 2D or 3D, but got ndims of {ndims}.')

    if isinstance(par, (list, tuple)):
        par = torch.stack([_to_tensor(p, dtype=torch.float32) for p in par], dim=-1)
    else:
        par = _to_tensor(par, dtype=torch.float32)

    if par.ndim == 0:
        par = par.reshape(1)

    num_shift = ndims
    num_rot = 1 if ndims == 2 else 3
    num_scale = ndims
    num_shear = 1 if ndims == 2 else 3
    total = num_shift + num_rot + num_scale + num_shear

    if par.shape[-1] > total:
        raise ValueError(f'Number of params exceeds value {total} expected for dimensionality.')

    sections = []
    cursor = 0
    spec = (
        (num_shift, 0.0),
        (num_rot, 0.0),
        (num_scale, 0.0 if shift_scale else 1.0),
        (num_shear, 0.0),
    )

    for size, default in spec:
        available = max(0, min(size, par.shape[-1] - cursor))
        if available > 0:
            segment = par[..., cursor:cursor + available]
        else:
            segment = par[..., :0]

        if available < size:
            pad_shape = list(par.shape[:-1]) + [size - available]
            padding = segment.new_full(pad_shape, default)
            segment = torch.cat((segment, padding), dim=-1) if available > 0 else padding

        sections.append(segment)
        cursor += size

    params = torch.cat(sections, dim=-1)

    shift = params[..., :num_shift]
    rot = params[..., num_shift:num_shift + num_rot]
    scale = params[..., num_shift + num_rot:num_shift + num_rot + num_scale]
    shear = params[..., -num_shear:]

    if shift_scale:
        scale = scale + 1.0

    shear_parts = torch.split(shear, 1, dim=-1)
    one = torch.ones_like(scale[..., :1])
    zero = torch.zeros_like(scale[..., :1])

    if ndims == 2:
        mat_shear = torch.stack((
            torch.cat((one, shear_parts[0]), dim=-1),
            torch.cat((zero, one), dim=-1),
        ), dim=-2)
    else:
        mat_shear = torch.stack((
            torch.cat((one, shear_parts[0], shear_parts[1]), dim=-1),
            torch.cat((zero, one, shear_parts[2]), dim=-1),
            torch.cat((zero, zero, one), dim=-1),
        ), dim=-2)

    mat_scale = torch.diag_embed(scale)
    mat_rot = angles_to_rotation_matrix(rot, deg=deg, ndims=ndims)
    out = torch.matmul(mat_rot, torch.matmul(mat_scale, mat_shear))

    shift_col = shift.unsqueeze(-1)
    out = torch.cat((out, shift_col), dim=-1)

    if last_row:
        batch_shape = out.shape[:-2]
        zeros = out.new_zeros(batch_shape + (1, ndims))
        ones = out.new_ones(batch_shape + (1, 1))
        row = torch.cat((zeros, ones), dim=-1)
        out = torch.cat((out, row), dim=-2)

    return out.squeeze(0) if par.ndim == 1 else out


def rotation_matrix_to_angles(mat, deg=True):
    """Compute Euler angles from an N-dimensional rotation matrix.

    We apply right-handed intrinsic rotations as R = X @ Y @ Z, where X, Y,
    and Z are matrices describing rotations about the x, y, and z-axis,
    respectively (see angles_to_rotation_matrix). Labeling these axes with
    indices 1-3 in the 3D case, we decompose the matrix

            [            c2*c3,             −c2*s3,      s2]
        R = [ s1*s2*c3 + c1*s3,  −s1*s2*s3 + c1*c3,  −s1*c2],
            [−c1*s2*c3 + s1*s3,   c1*s2*s3 + s1*c3,   c1*c2]

    where si and ci are the sine and cosine of the angle of rotation about
    axis i. When the angle of rotation about the y-axis is 90 or -90 degrees,
    the system loses one degree of freedom, and the solution is not unique.
    In this gimbal lock case, we set the angle `ang[0]` to zero and solve for
    `ang[2]`.

    Arguments:
        mat: Array-like input matrix to derive rotation angles from, of shape
            (..., N, N + 1) or (..., N + 1, N + 1), where N is 2 or 3.
        deg: Return rotation angles in degrees instead of radians.

    Returns:
        ang: Tensor of shape (..., M) holding the derived rotation angles. The
        size M of the right-most dimension is 3 in 3D and 1 in 2D.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    tensor = _to_tensor(mat, dtype=torch.float32)
    num_dim = tensor.shape[-1]
    if num_dim not in (2, 3) or tensor.shape[-2] != num_dim:
        raise ValueError('rotation_matrix_to_angles expects square 2D or 3D matrices')

    clip = lambda x: torch.clamp(x, -1.0, 1.0)

    if num_dim == 2:
        y = clip(tensor[..., 1, 0])
        x = clip(tensor[..., 0, 0])
        ang = torch.atan2(y, x).unsqueeze(-1)
    else:
        ang2 = torch.asin(clip(tensor[..., 0, 2]))

        ang1_a = torch.zeros_like(ang2)
        ang3_a = torch.atan2(clip(tensor[..., 1, 0]), clip(tensor[..., 1, 1]))

        c2 = torch.cos(ang2)
        eps = torch.finfo(tensor.dtype).eps

        safe_div = lambda num, denom, default: torch.where(
            torch.abs(denom) > eps,
            num / denom,
            default,
        )

        y1 = safe_div(-tensor[..., 1, 2], c2, torch.zeros_like(c2))
        x1 = safe_div(tensor[..., 2, 2], c2, torch.ones_like(c2))
        ang1_b = torch.atan2(clip(y1), clip(x1))

        y3 = safe_div(-tensor[..., 0, 1], c2, torch.zeros_like(c2))
        x3 = safe_div(tensor[..., 0, 0], c2, torch.ones_like(c2))
        ang3_b = torch.atan2(clip(y3), clip(x3))

        is_case = torch.abs(torch.abs(ang2) - 0.5 * math.pi) < 1e-6
        ang1 = torch.where(is_case, ang1_a, ang1_b)
        ang3 = torch.where(is_case, ang3_a, ang3_b)
        ang = torch.stack((ang1, ang2, ang3), dim=-1)

    if deg:
        ang = ang * (180.0 / math.pi)
    return ang


def affine_matrix_to_params(mat, deg=True):
    """Derive affine parameters from an N-dimensional transformation matrix.

    The affine transform operates in a right-handed frame of reference, with
    right-handed intrinsic rotations (see params_to_affine_matrix).

    Arguments:
        mat: Array-like input matrix to derive affine parameters from, of
            shape (..., N, N + 1) or (..., N + 1, N + 1), as the last row
            always is ``(*[0] * N, 1)``. N can be 2 or 3.
        deg: Return rotation angles in degrees instead of radians.

    Returns:
        par: Tensor of shape (..., K) holding the affine parameters derived
            from `mat`, where K is ``N * (N + 1)``. The parameters along the
            last axis represent, in order: translation, rotation, scaling, and
            shear. In 3D, for example, the first three indices specify
            translations along the x, y, and z-axis, and similarly for the
            remaining indices.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    tensor = _to_tensor(mat, dtype=torch.float32)

    num_dim = tensor.shape[-1] - 1
    if num_dim not in (2, 3) or tensor.shape[-2] - num_dim not in (0, 1):
        raise ValueError(f'invalid affine shape {tensor.shape}')

    shift = tensor[..., :num_dim, -1]
    matrix = tensor[..., :num_dim, :num_dim]

    lower = torch.linalg.cholesky(matrix.transpose(-1, -2) @ matrix)
    scale = torch.diagonal(lower, dim1=-2, dim2=-1)
    det_sign = torch.sign(torch.linalg.det(matrix))
    scale0 = scale[..., 0] * det_sign
    scale = torch.cat((scale0.unsqueeze(-1), scale[..., 1:]), dim=-1)

    strip = torch.diag_embed(scale)
    upper = lower.transpose(-1, -2)
    upper = torch.linalg.solve(strip, upper)

    if num_dim == 2:
        shear = upper[..., 0, 1].unsqueeze(-1)
    else:
        shear = torch.stack((upper[..., 0, 1], upper[..., 0, 2], upper[..., 1, 2]), dim=-1)

    zero = scale.new_zeros(scale.shape[:-1] + ((num_dim - 1) * 3,))
    par = torch.cat((zero, scale, shear), dim=-1)
    strip = params_to_affine_matrix(par, ndims=num_dim)[..., :-1]
    rot_matrix = torch.matmul(matrix, torch.linalg.inv(strip))
    rot = rotation_matrix_to_angles(rot_matrix, deg=deg)

    return torch.cat((shift, rot, scale, shear), dim=-1)


def fit_affine(x_source, x_target, weights=None):
    """Fit an affine transform between two sets of corresponding points.

    Fit an N-dimensional affine transform between two sets of M corresponding
    points in an ordinary or weighted least-squares sense. Note that when
    working with images, source coordinates correspond to the target image and
    vice versa.

    Arguments:
        x_source: Array-like source coordinates of shape (..., M, N).
        x_target: Array-like target coordinates of shape (..., M, N).
        weights: Optional array-like weights of shape (..., M) or (..., M, 1).

    Returns:
        mat: Affine transformation matrix of shape (..., N, N + 1), fitted such
            that ``x_t = mat[..., :-1] @ x_s + mat[..., -1:]`` with
            ``x_s = x_t.transpose(-1, -2)``. The last row of `mat` is omitted as it is always
            ``(*[0] * N, 1)``.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    src = _to_tensor(x_source, dtype=torch.float32)
    trg = _to_tensor(x_target, dtype=src.dtype, device=src.device)

    ones = torch.ones(trg.shape[:-1] + (1,), dtype=trg.dtype, device=trg.device)
    x = torch.cat((trg, ones), dim=-1)
    x_transp = x.transpose(-1, -2)
    y = src

    if weights is not None:
        w = _to_tensor(weights, dtype=trg.dtype, device=trg.device)
        if w.ndim == x.ndim:
            w = w[..., 0]
        x_transp = x_transp * w.unsqueeze(-2)

    beta = torch.linalg.solve(x_transp @ x, x_transp @ y)
    return beta.transpose(-1, -2)
