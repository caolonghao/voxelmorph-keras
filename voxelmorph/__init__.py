# ---- voxelmorph ----
# unsupervised learning for image registration

import os

# set version
__version__ = '0.2'


from packaging import version

# ensure valid neurite version is available
import neurite
minv = '0.2'
curv = getattr(neurite, '__version__', None)

if curv is None or version.parse(curv) < version.parse(minv):
    raise ImportError(f'voxelmorph requires neurite version {minv} or greater, '
                      f'but found version {curv}')

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError('voxelmorph now targets the PyTorch-backed Keras runtime; '
                      'install torch to continue.') from exc

minv = '2.0'
curv = getattr(torch, '__version__', None)
if curv is None or version.parse(curv) < version.parse(minv):
    raise ImportError(f'voxelmorph requires torch version {minv} or greater, '
                      f'but found version {curv}')


from . import py
from . import utils
from . import generators
from . import layers
from . import losses
from . import networks
