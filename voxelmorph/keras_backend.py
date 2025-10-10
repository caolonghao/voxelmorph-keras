"""Compatibility layer exposing a minimal TensorFlow-like API backed by Keras ops.

This module lets the rest of Voxelmorph call `tf.*` functions while we migrate to
Keras 3. Only the TensorFlow symbols that were used in the original code base are
implemented here. Missing functionality should be added on demand.
"""

from __future__ import annotations

import math
import time
import types
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - optional dependency when running on non-Torch backends
    import torch
except Exception:  # pragma: no cover
    torch = None

import keras
from keras import ops
from keras import random as keras_random

# -----------------------------------------------------------------------------
# DTypes
# -----------------------------------------------------------------------------

float16 = "float16"
float32 = "float32"
float64 = "float64"
int16 = "int16"
int32 = "int32"
int64 = "int64"
__version__ = keras.__version__


class _DType:
    def __init__(self, name: str, np_dtype: Optional[np.dtype] = None):
        self.name = name
        if np_dtype is None:
            try:
                np_dtype = np.dtype(name)
            except Exception:  # pragma: no cover
                if name == 'bfloat16':
                    np_dtype = np.dtype('float32')
                else:
                    np_dtype = np.dtype('float32') if name.startswith('float') else np.dtype('int32')
        self._np_dtype = np_dtype

    @property
    def as_numpy_dtype(self) -> np.dtype:
        return self._np_dtype

    @property
    def is_floating(self) -> bool:
        return np.issubdtype(self._np_dtype, np.floating)

    @property
    def is_integer(self) -> bool:
        return np.issubdtype(self._np_dtype, np.integer)

    def __str__(self) -> str:  # pragma: no cover
        return self.name

    def __repr__(self) -> str:  # pragma: no cover
        return f"_DType(name={self.name!r})"


class _DTypes:
    @staticmethod
    def as_dtype(value: Union[str, np.dtype, None, Any]) -> Optional[_DType]:
        if value is None:
            return None
        if isinstance(value, str):
            short = value.split('.', 1)[-1]
            try:
                return _DType(name=np.dtype(short).name)
            except Exception:  # pragma: no cover
                return _DType(name=short)
        if isinstance(value, np.dtype):
            return _DType(name=value.name, np_dtype=value)
        if torch is not None and isinstance(value, torch.dtype):
            name = str(value).split('.', 1)[-1]
            try:
                return _DType(name=np.dtype(name).name)
            except Exception:  # pragma: no cover
                return _DType(name=name)
        name = getattr(value, "name", None)
        if isinstance(name, str) and name:
            try:
                return _DType(name=np.dtype(name).name)
            except Exception:  # pragma: no cover
                return _DType(name=name)
        try:
            return _DType(name=np.dtype(value).name)
        except Exception:  # pragma: no cover
            return _DType(name=str(value))


dtypes = _DTypes()


# -----------------------------------------------------------------------------
# Basic tensor creation / manipulation
# -----------------------------------------------------------------------------

TensorLike = Any


def convert_to_tensor(value: Any, dtype: Optional[str] = None) -> TensorLike:
    dtype_obj = dtypes.as_dtype(dtype)
    dtype_name = dtype_obj.name if isinstance(dtype_obj, _DType) else dtype_obj
    return ops.convert_to_tensor(value, dtype=dtype_name)


def constant(value: Any, dtype: Optional[str] = None) -> TensorLike:
    dtype_obj = dtypes.as_dtype(dtype)
    dtype_name = dtype_obj.name if isinstance(dtype_obj, _DType) else dtype_obj
    return ops.convert_to_tensor(value, dtype=dtype_name)


as_tensor = convert_to_tensor


def cast(x: TensorLike, dtype: str) -> TensorLike:
    dtype_obj = dtypes.as_dtype(dtype)
    dtype_name = dtype_obj.name if isinstance(dtype_obj, _DType) else dtype_obj
    return ops.cast(x, dtype_name)


def reshape(x: TensorLike, shape: Sequence[int]) -> TensorLike:
    return ops.reshape(x, shape)


def expand_dims(x: TensorLike, axis: int = 0) -> TensorLike:
    return ops.expand_dims(x, axis)


def squeeze(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None) -> TensorLike:
    return ops.squeeze(x, axis)


def transpose(x: TensorLike, perm: Optional[Sequence[int]] = None) -> TensorLike:
    return ops.transpose(x, perm)


def shape(x: TensorLike) -> TensorLike:
    return ops.shape(x)


def rank(x: TensorLike) -> int:
    # Prefer static rank when available to avoid tracing ops when unnecessary.
    if hasattr(x, "shape") and x.shape is not None:
        return len(x.shape)
    return ops.shape(x).shape[0]


def size(x: TensorLike) -> TensorLike:
    return ops.size(x)


def stack(values: Sequence[TensorLike], axis: int = 0) -> TensorLike:
    return ops.stack(values, axis=axis)


def concat(values: Sequence[TensorLike], axis: int) -> TensorLike:
    tensors = [ops.convert_to_tensor(v) if not hasattr(v, "shape") else v for v in values]
    return ops.concatenate(tensors, axis=axis)


def pad(x: TensorLike, paddings: TensorLike, mode: str = "constant", constant_values: float = 0.0) -> TensorLike:
    return ops.pad(x, paddings, mode=mode, constant_values=constant_values)


def tile(x: TensorLike, multiples: Sequence[int]) -> TensorLike:
    return ops.tile(x, multiples)


def broadcast_to(x: TensorLike, shape: Sequence[int]) -> TensorLike:
    return ops.broadcast_to(x, shape)


def gather(params: TensorLike, indices: TensorLike, axis: int = 0) -> TensorLike:
    return ops.take(params, indices, axis=axis)


def split(value: TensorLike, num_or_size_splits, axis: int = 0):
    return ops.split(value, num_or_size_splits, axis=axis)


def clip_by_value(x: TensorLike, clip_value_min: float, clip_value_max: float) -> TensorLike:
    return ops.clip(x, clip_value_min, clip_value_max)


# -----------------------------------------------------------------------------
# Creation helpers
# -----------------------------------------------------------------------------


def zeros(shape: Sequence[int], dtype: str = float32) -> TensorLike:
    return ops.zeros(shape, dtype=dtype)


def zeros_like(x: TensorLike, dtype: Optional[str] = None) -> TensorLike:
    return ops.zeros_like(x, dtype=dtype)


def ones(shape: Sequence[int], dtype: str = float32) -> TensorLike:
    return ops.ones(shape, dtype=dtype)


def ones_like(x: TensorLike, dtype: Optional[str] = None) -> TensorLike:
    return ops.ones_like(x, dtype=dtype)


def eye(num_rows: int,
        num_cols: Optional[int] = None,
        dtype: str = float32,
        batch_shape: Optional[Sequence[int]] = None,
        k: int = 0) -> TensorLike:
    matrix = ops.eye(num_rows, M=num_cols, k=k, dtype=dtype)
    if batch_shape:
        result = matrix
        for _ in batch_shape:
            result = ops.expand_dims(result, axis=0)
        tile_multipliers = tuple(batch_shape) + (1,) * ops.ndim(matrix)
        result = ops.tile(result, tile_multipliers)
        return result
    return matrix


def range(start: Union[int, float],
          limit: Optional[Union[int, float]] = None,
          delta: Union[int, float] = 1,
          dtype: Optional[str] = None) -> TensorLike:
    if limit is None:
        start, limit = 0, start
    return ops.arange(start, stop=limit, step=delta, dtype=dtype)


linspace = ops.linspace


def meshgrid(*x: TensorLike, indexing: str = "ij") -> Sequence[TensorLike]:
    return ops.meshgrid(*x, indexing=indexing)


# -----------------------------------------------------------------------------
# Math ops
# -----------------------------------------------------------------------------

abs = ops.abs
add = ops.add


def add_n(values: Sequence[TensorLike]) -> TensorLike:
    return ops.sum(ops.stack(list(values), axis=0), axis=0)

maximum = ops.maximum
minimum = ops.minimum
multiply = ops.multiply
subtract = ops.subtract
pow = ops.power
square = ops.square
sqrt = ops.sqrt
sin = ops.sin
cos = ops.cos
asin = ops.arcsin
atan = ops.arctan
atan2 = ops.arctan2
sign = ops.sign

reduce_sum = ops.sum
reduce_mean = ops.mean
reduce_max = ops.max
reduce_min = ops.min
reduce_any = ops.any

log = ops.log
exp = ops.exp
floor = ops.floor
ceil = ops.ceil
round = ops.round


def reduce_variance(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> TensorLike:
    mean = ops.mean(x, axis=axis, keepdims=True)
    squared = ops.square(ops.subtract(x, mean))
    return ops.mean(squared, axis=axis, keepdims=keepdims)


def reduce_std(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> TensorLike:
    variance = reduce_variance(x, axis=axis, keepdims=keepdims)
    return ops.sqrt(variance)


def reduce_prod(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> TensorLike:
    if axis is None:
        axis = tuple(range(rank(x)))
    return ops.prod(x, axis=axis, keepdims=keepdims)


def maximum_with_epsilon(x: TensorLike, eps: float) -> TensorLike:
    return ops.maximum(x, eps)


def divide_no_nan(x: TensorLike, y: TensorLike) -> TensorLike:
    zero = ops.zeros_like(y)
    cond = ops.equal(y, zero)
    safe_y = ops.where(cond, ops.ones_like(y), y)
    result = ops.divide(x, safe_y)
    return ops.where(cond, zero, result)


def where(condition: TensorLike, x: TensorLike, y: TensorLike) -> TensorLike:
    return ops.where(condition, x, y)


# Comparison helpers

greater = ops.greater
less = ops.less
less_equal = ops.less_equal
equal = ops.equal
logical_and = ops.logical_and
logical_or = ops.logical_or
logical_not = ops.logical_not


# -----------------------------------------------------------------------------
# Linear algebra
# -----------------------------------------------------------------------------

matmul = ops.matmul
batch_matmul = ops.matmul
matrix_transpose = ops.transpose


class _Linalg(types.SimpleNamespace):
    def __init__(self):
        eigvalsh = getattr(ops.linalg, "eigvalsh", None)
        if eigvalsh is None:
            # Keras 3.0 does not expose eigvalsh; synthesize it from eigh.
            def eigvalsh(x: TensorLike) -> TensorLike:
                eigenvalues, _ = ops.linalg.eigh(x)
                return eigenvalues

        super().__init__(
            det=ops.linalg.det,
            inv=ops.linalg.inv,
            cholesky=ops.linalg.cholesky,
            norm=ops.linalg.norm,
            eigh=ops.linalg.eigh,
            eigvalsh=eigvalsh,
            matrix_transpose=ops.transpose,
            diag=ops.diag,
            diag_part=lambda x: ops.diagonal(x, axis1=-2, axis2=-1),
            matmul=ops.matmul,
        )

    def sqrtm(self, x: TensorLike) -> TensorLike:
        evals, evecs = ops.linalg.eigh(x)
        evals = ops.maximum(evals, 0.0)
        sqrt_evals = ops.sqrt(evals)
        diag = ops.diag(sqrt_evals)
        return evecs @ diag @ ops.transpose(evecs)


linalg = _Linalg()


# -----------------------------------------------------------------------------
# Random helpers
# -----------------------------------------------------------------------------


class _RandomGenerator:
    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self._rng = keras_random.RandomGenerator(seed)

    @classmethod
    def from_non_deterministic_state(cls):
        return cls(int(time.time_ns() % (2 ** 31)))

    def reset_from_seed(self, seed: int) -> None:
        self._seed = seed
        self._rng = keras_random.RandomGenerator(seed)

    def normal(self, shape: Sequence[int], mean: float = 0.0, stddev: float = 1.0, dtype: str = float32) -> TensorLike:
        return self._rng.normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype)

    def truncated_normal(self, shape: Sequence[int], mean: float = 0.0, stddev: float = 1.0, dtype: str = float32) -> TensorLike:
        # RandomGenerator currently exposes truncated_normal via distribution class
        return keras_random.TruncatedNormal(mean=mean, stddev=stddev, seed=self._seed)(shape, dtype=dtype)

    def uniform(self, shape: Sequence[int], minval: float = 0.0, maxval: float = 1.0, dtype: str = float32) -> TensorLike:
        return self._rng.uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype)

    def shuffle(self, value: TensorLike, axis: int = 0) -> TensorLike:
        return self._rng.shuffle(value, axis=axis)


class _Random(types.SimpleNamespace):
    Generator = _RandomGenerator

    def normal(self, shape: Sequence[int], mean: float = 0.0, stddev: float = 1.0, dtype: str = float32, seed: Optional[int] = None) -> TensorLike:
        return _RandomGenerator(seed).normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype)

    def truncated_normal(self, shape: Sequence[int], mean: float = 0.0, stddev: float = 1.0, dtype: str = float32, seed: Optional[int] = None) -> TensorLike:
        return _RandomGenerator(seed).truncated_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype)

    def uniform(self, shape: Sequence[int], minval: float = 0.0, maxval: float = 1.0, dtype: str = float32, seed: Optional[int] = None) -> TensorLike:
        return _RandomGenerator(seed).uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype)

    def shuffle(self, value: TensorLike, axis: int = 0, seed: Optional[int] = None) -> TensorLike:
        return _RandomGenerator(seed).shuffle(value, axis=axis)


random = _Random()


# -----------------------------------------------------------------------------
# Higher-order ops
# -----------------------------------------------------------------------------


def map_fn(fn: Callable[[TensorLike], TensorLike], elems: TensorLike, dtype: Optional[str] = None, fn_output_signature: Optional[str] = None) -> TensorLike:
    if isinstance(elems, (list, tuple)):
        elems = tuple(elems)

        def _body(index):
            items = [ops.take(el, index, axis=0) for el in elems]
            return fn(items)

        num = ops.shape(elems[0])[0]
        indices = ops.arange(0, num)
        return ops.map(_body, indices)

    try:
        return ops.map(fn, elems)
    except TypeError:
        return ops.map(fn, elems)


def vectorized_map(fn: Callable[[TensorLike], TensorLike], elems: TensorLike) -> TensorLike:
    if hasattr(ops, "vectorized_map"):
        return ops.vectorized_map(fn, elems)
    return map_fn(fn, elems)


# -----------------------------------------------------------------------------
# Convolution helpers (used by NCC losses)
# -----------------------------------------------------------------------------


def _conv_nd(ndims: int, x: TensorLike, filt: TensorLike, strides: Union[int, Sequence[int]], padding: str) -> TensorLike:
    if isinstance(strides, int):
        strides = (1,) * ndims
    return ops.conv(x, filt, strides, padding.upper(), data_format="channels_last")


class _NN(types.SimpleNamespace):
    def __getattr__(self, name: str) -> Callable:
        if name.startswith("conv") and name.endswith("d"):
            nd = int(name[4:-1])
            return lambda x, filt, strides, padding: _conv_nd(nd, x, filt, strides, padding)
        raise AttributeError(name)


nn = _NN()


def _normalize_axes(axes: Optional[Union[int, Sequence[int]]], ndim: int) -> Tuple[int, ...]:
    if axes is None:
        return tuple(range(ndim))
    if isinstance(axes, int):
        axes = (axes,)
    normalized = []
    for axis in axes:
        normalized.append(axis if axis >= 0 else ndim + axis)
    return tuple(normalized)


class _Signal(types.SimpleNamespace):
    def __init__(self):
        def _shift(direction: int) -> Callable[[TensorLike, Optional[Union[int, Sequence[int]]]], TensorLike]:
            def _fn(x: TensorLike, axes: Optional[Union[int, Sequence[int]]] = None) -> TensorLike:
                tensor = ops.convert_to_tensor(x)
                ndim = len(tensor.shape)
                target_axes = _normalize_axes(axes, ndim)
                result = tensor
                for axis in target_axes:
                    static_size = result.shape[axis]
                    if static_size is not None:
                        shift_val: Union[int, TensorLike] = static_size // 2
                    else:
                        dyn_shape = ops.shape(result)
                        size = dyn_shape[axis]
                        size_f = ops.cast(size, "float32")
                        half = ops.floor(ops.divide(size_f, ops.convert_to_tensor(2.0, dtype="float32")))
                        shift_val = ops.cast(half, "int32")
                    if direction < 0:
                        if isinstance(shift_val, int):
                            shift_val = -shift_val
                        else:
                            shift_val = ops.negative(shift_val)
                    result = ops.roll(result, shift_val, axis)
                return result

            return _fn

        def _missing(name: str) -> Callable[..., Any]:
            def _fn(*_args, **_kwargs):  # pragma: no cover
                raise NotImplementedError(
                    f"Signal operation '{name}' is not implemented without TensorFlow."
                )

            return _fn

        super().__init__(
            fft=_missing("fft"),
            ifft=_missing("ifft"),
            fftshift=_shift(+1),
            ifftshift=_shift(-1),
        )


signal = _Signal()


# -----------------------------------------------------------------------------
# Utility stubs
# -----------------------------------------------------------------------------


class _Math(types.SimpleNamespace):
    divide_no_nan = staticmethod(divide_no_nan)
    log = staticmethod(log)
    exp = staticmethod(exp)
    reduce_std = staticmethod(reduce_std)
    reduce_variance = staticmethod(reduce_variance)


math = _Math()
div_no_nan = divide_no_nan


class _Config:
    def set_soft_device_placement(self, _value: bool):
        pass

    def list_physical_devices(self, *_args, **_kwargs):
        return []

    class experimental:
        @staticmethod
        def set_memory_growth(*_args, **_kwargs):
            pass


config = _Config()




class _Debugging:
    @staticmethod
    def assert_equal(x: TensorLike, y: TensorLike, message: Optional[str] = None):
        try:
            if np.array_equal(np.array(x), np.array(y)):
                return
        except Exception:  # pragma: no cover
            return
        raise AssertionError(message or f"Assertion failed: {x} != {y}")

debugging = _Debugging()

class _Utils(types.SimpleNamespace):
    @staticmethod
    def get_file(*args, **kwargs):
        from keras.utils import get_file
        return get_file(*args, **kwargs)


utils = _Utils()


# -----------------------------------------------------------------------------
# Misc helpers used throughout code base
# -----------------------------------------------------------------------------


def is_tensor(x: Any) -> bool:
    return hasattr(x, "shape")


class TensorShape(tuple):
    """Minimal TensorShape placeholder used for compatibility checks."""

    def as_list(self):
        return list(self)


class _Compat:
    def __init__(self):
        experimental = types.SimpleNamespace(output_all_intermediates=lambda *args, **kwargs: None)
        self.v1 = types.SimpleNamespace(
            Dimension=int,
            disable_eager_execution=lambda: None,
            experimental=experimental,
        )


compat = _Compat()


Session = None
ConfigProto = None
newaxis = None


def device(*_args, **_kwargs):
    return types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: False)()


def placeholder(dtype: str, shape: Sequence[int], name: Optional[str] = None):
    return keras.Input(shape=shape, dtype=dtype, name=name)


def stack_along_batch(*tensors: TensorLike) -> TensorLike:
    return ops.stack(tensors, axis=0)


__all__ = [name for name in globals() if not name.startswith("_")]
