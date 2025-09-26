# third party imports
import numpy as np
import torch

# local imports
import neurite as ne
from . import utils

def _resolve_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        mapping = {
            'float16': torch.float16,
            'float32': torch.float32,
            'float64': torch.float64,
        }
        return mapping.get(dtype.lower(), torch.float32)
    try:
        return torch.as_tensor([], dtype=dtype).dtype
    except TypeError:
        return torch.float32


def _make_generator(seed=None, device='cpu'):
    if seed is None:
        return torch.Generator(device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def draw_flip_matrix(grid_shape, shift_center=True, last_row=True, dtype=torch.float32, seed=None):
    """
    Draw a matrix transform that randomly flips axes of N-dimensional space.

    Parameters:
        grid_shape: Spatial shape of the image in voxels, exluding batches and features.
        shift_center: Whether zero is at the center of the grid. Should be identical to the value.
            used for vxm.utils.affine_to_shift or vxm.layers.SpatialTransformer.
        last_row: Append the last row of the transform to return a square matrix.
        dtype: Floating-point output data type.
        seed: Integer for reproducible randomization.

    Returns:
        Flip matrix of shape (M, N + 1), where M is N or N + 1, depending on `last_row`.

    If you find this function useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """
    dtype = _resolve_dtype(dtype)
    ndims = len(grid_shape)
    device = 'cpu'
    generator = _make_generator(seed, device=device)

    grid_shape = torch.as_tensor(grid_shape, dtype=dtype, device=device)
    rand_bit = torch.randn((ndims,), generator=generator, device=device) > 0
    diag = torch.ones(ndims, dtype=dtype, device=device)
    diag[rand_bit] = -1.0
    diag_mat = torch.diag(diag)

    shift = ((grid_shape - 1.0) * rand_bit.to(dtype)).unsqueeze(-1)
    if shift_center:
        shift = torch.zeros((ndims, 1), dtype=dtype, device=device)

    out = torch.cat((diag_mat, shift), dim=1)
    if last_row:
        row = torch.zeros((1, ndims + 1), dtype=dtype, device=device)
        row[0, -1] = 1.0
        out = torch.cat((out, row), dim=0)

    return out


def draw_swap_matrix(ndims, last_row=True, dtype=torch.float32, seed=None):
    """
    Draw a matrix transform that randomly swaps axes of N-dimensional space.

    Parameters:
        ndims: Number of spatial dimensions, excluding batches and features.
        last_row: Append the last row of the transform to return a square matrix.
        dtype: Floating-point output data type.
        seed: Integer for reproducible randomization.

    Returns:
        Swap matrix of shape (M, N + 1), where M is N or N + 1, depending on `last_row`.

    If you find this function useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """
    dtype = _resolve_dtype(dtype)
    device = 'cpu'
    generator = _make_generator(seed, device=device)

    mat = torch.eye(ndims, ndims + 1, dtype=dtype, device=device)
    perm = torch.randperm(ndims, generator=generator, device=device)
    mat = mat[perm]

    if last_row:
        row = torch.zeros((1, ndims + 1), dtype=dtype, device=device)
        row[0, -1] = 1.0
        mat = torch.cat((mat, row), dim=0)

    return mat


def draw_affine_params(shift=None,
                       rot=None,
                       scale=None,
                       shear=None,
                       normal_shift=False,
                       normal_rot=False,
                       normal_scale=False,
                       normal_shear=False,
                       shift_scale=False,
                       ndims=3,
                       batch_shape=None,
                       concat=True,
                       dtype=torch.float32,
                       seeds={},
                       device=None):
    """
    Draw translation, rotation, scaling and shearing parameters defining an affine transform in
    N-dimensional space, where N is 2 or 3. Choose parameters wisely: there is no check for
    negative or zero scaling!

    Parameters:
        shift: Translation sampling range x around identity. Values will be sampled uniformly from
            [-x, x]. When sampling from a normal distribution, x is the standard deviation (SD).
            The same x will be used for each dimension, unless an iterable of length N is passed,
            specifying a value separately for each axis. None means 0.
        rot: Rotation sampling range (see `shift`). Accepts only one value in 2D.
        scale: Scaling sampling range x. Parameters will be sampled around identity as for `shift`,
            unless `shift_scale` is set. When sampling normally, scaling parameters will be
            truncated beyond two standard deviations.
        shear: Shear sampling range (see `shift`). Accepts only one value in 2D.
        normal_shift: Sample translations normally rather than uniformly.
        normal_rot: See `normal_shift`.
        normal_scale: Draw scaling parameters from a normal distribution, truncating beyond 2 SDs.
        normal_shear: See `normal_shift`.
        shift_scale: Add 1 to any drawn scaling parameter When sampling uniformly, this will
            result in scaling parameters falling in [1 - x, 1 + x] instead of [-x, x].
        ndims: Number of dimensions. Must be 2 or 3.
        normal: Sample parameters normally instead of uniformly.
        batch_shape: A list or tuple. If provided, the output will have leading batch dimensions.
        concat: Concatenate the output along the last axis to return a single tensor.
        dtype: Floating-point output data type.
        seeds: Dictionary of integers for reproducible randomization. Keywords must be in ('shift',
            'rot', 'scale', 'shear').

    Returns:
        A tuple of tensors with shapes (..., N), (..., M), (..., N), and (..., M) defining
        translation, rotation, scaling, and shear, respectively, where M is 3 in 3D and 1 in 2D.
        With `concat=True`, the function will concatenate the output along the last dimension.

    See also:
        vxm.layers.DrawAffineParams
        vxm.layers.ParamsToAffineMatrix
        vxm.utils.params_to_affine_matrix

    If you find this function useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """
    assert ndims in (2, 3), 'only 2D and 3D supported'
    n = 1 if ndims == 2 else 3
    dtype = _resolve_dtype(dtype)
    device = torch.device(device) if device is not None else torch.device('cpu')

    # Look-up tables.
    splits = dict(shift=ndims, rot=n, scale=ndims, shear=n)
    inputs = dict(shift=shift, rot=rot, scale=scale, shear=shear)
    trunc = dict(shift=False, rot=False, scale=True, shear=False)
    normal = dict(shift=normal_shift, rot=normal_rot, scale=normal_scale, shear=normal_shear)

    # Normalize inputs.
    shapes = {}
    ranges = {}
    for k, n in splits.items():
        x = np.ravel(0 if inputs[k] is None else inputs[k])
        if len(x) == 1:
            x = np.repeat(x, repeats=n)
        assert len(x) == n, f'unexpected number of parameters {len(x)} ({k})'
        ranges[k] = x
        if batch_shape is None:
            shapes[k] = (n,)
        else:
            if isinstance(batch_shape, torch.Tensor):
                shape_vals = tuple(int(v) for v in batch_shape.to(device='cpu').tolist())
            else:
                shape_vals = tuple(int(v) for v in np.atleast_1d(batch_shape))
            shapes[k] = shape_vals + (n,)

    # Choose distribution.
    def sample(lim, shape, normal, trunc, seed):
        lim = torch.as_tensor(lim, dtype=dtype, device=device)
        generator = _make_generator(seed, device=device)
        expand = (1,) * (len(shape) - 1) + (lim.shape[0],)
        lim_view = lim.view(expand)

        if normal:
            out = torch.randn(shape, generator=generator, dtype=dtype, device=device) * lim_view
            if trunc:
                limit = 2 * lim_view
                out = torch.clamp(out, -limit, limit)
        else:
            out = torch.rand(shape, generator=generator, dtype=dtype, device=device) * 2 - 1
            out = out * lim_view
        return out

    # Sample parameters.
    par = {}
    seeds = seeds.copy()
    for k, lim in ranges.items():
        par[k] = sample(lim, shapes[k], normal[k], trunc[k], seed=seeds.pop(k, None))
    if shift_scale:
        par['scale'] = par['scale'] + 1
    assert not seeds, f'unknown seeds {seeds}'

    # Output.
    order = ('shift', 'rot', 'scale', 'shear')
    out = tuple(par[k] for k in order)
    return torch.cat(out, dim=-1) if concat else out


def down_up_sample(x,
                   stride_min=1,
                   stride_max=8,
                   axes=None,
                   prob=1,
                   interp_method='linear',
                   rand=None):
    """
    Symmetrically downsample a tensor by a factor f (stride) using
    nearest-neighbor interpolation and upsample again, to reduce its
    resolution. Both f and the downsampling axes can be randomized. The
    function does not bother with anti-aliasing, as it is intended for
    augmentation after a random blurring step.

    Parameters:
        x: Input tensor or NumPy array of shape (*spatial, channels).
        stride_min: Minimum downsampling factor.
        stride_max: Maximum downsampling factor.
        axes: Spatial axes to draw the downsampling axis from. None means all axes.
        prob: Downsampling probability. A value of 1 means always, 0 never.
        interp_method: Upsampling method. Choose 'linear' or 'nearest'.
        rand: Random generator. Initialize externally for graph building.

    Returns:
        Tensor with reduced resolution.

    Notes:
        This function differs from ne.utils.subsample in that it downsamples
        along several axes, always restores the shape of the input tensor, and
        moves the image content less.

    If you find this function useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """
    # Validate inputs.
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    ndim = x.ndim - 1
    size = x.shape[:-1]
    axes = ne.py.utils.normalize_axes(axes, size, none_means_all=True)
    dtype = x.dtype
    device = x.device
    if rand is None:
        rand = torch.Generator(device=device)

    # Draw thickness.
    assert 1 <= stride_min and stride_min <= stride_max, 'invalid strides'
    fact = torch.empty(ndim, dtype=dtype, device=device)
    fact.uniform_(float(stride_min), float(stride_max), generator=rand)

    # One-hot encode axes.
    axes_mask = torch.zeros(ndim, dtype=torch.bool, device=device)
    axes_mask[list(axes)] = True

    # Decide where to downsample.
    assert 0 <= prob <= 1, f'{prob} not a probability'
    bit = torch.rand(ndim, generator=rand, device=device) < prob
    fact = torch.where(bit & axes_mask, fact, torch.ones_like(fact))

    # Downsample. Always use nearest.
    diag = torch.cat((fact, torch.ones(1, dtype=dtype, device=device)))
    trans = torch.diag(diag)
    x = utils.transform(x, trans, interp_method='nearest')

    # Upsample.
    diag = torch.cat(((1.0 / fact), torch.ones(1, dtype=dtype, device=device)))
    trans = torch.diag(diag)
    x = utils.transform(x, trans, interp_method=interp_method)

    return x if x.dtype == dtype else x.to(dtype)
