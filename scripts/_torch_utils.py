"""Common helpers for torch-backed Voxelmorph CLI scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np


try:  # torch is an optional runtime dependency for CPU-only environments
    import torch
except ImportError:  # pragma: no cover - torch is required for GPU acceleration
    torch = None  # type: ignore


def setup_device(gpu: str | None) -> Tuple[str, int]:
    """Configure CUDA visibility based on the user-specified GPU string.

    Returns the device descriptor (``'cpu'`` or ``'cuda:i'``) and the number of
    visible devices. Multi-GPU execution is not yet supported; when multiple
    device IDs are provided we only expose the first one and warn the user.
    """

    if torch is None or not torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return 'cpu', 1

    if gpu is None or gpu == '':
        count = torch.cuda.device_count()
        if count == 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            return 'cpu', 1
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(count))
        return 'cuda:0', count

    if gpu == '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return 'cpu', 1

    device_ids = [idx.strip() for idx in gpu.split(',') if idx.strip()]
    if not device_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return 'cpu', 1

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(device_ids)
    primary = device_ids[0]
    return f'cuda:{primary}', 1


def ensure_directory(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_weights(model, path: str) -> None:
    """Save model weights in a backend-agnostic way."""

    if path.endswith('.h5') or path.endswith('.keras'):
        model.save_weights(path)
    else:
        model.save_weights(f'{path}.weights.h5')


def load_weights_if_available(model, path: str | None) -> None:
    if path:
        model.load_weights(path)


def numpy_collate(batch):
    """Stack a generator batch (list/tuple) into numpy arrays."""

    if isinstance(batch, (list, tuple)):
        return [np.asarray(item) for item in batch]
    return np.asarray(batch)

