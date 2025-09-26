# Quickstart: register two volumes

The goal of this walkthrough is to register a moving brain volume to a fixed atlas by reusing a pretrained VoxelMorph model. All commands assume you run them from the repository root (the directory containing `scripts/`).

## 1. Requirements
- Python 3.9+
- `torch>=2.0`, `torchvision`, and `neurite>=0.2`
- `voxelmorph` installed either from PyPI or the local checkout:
  - PyPI: `pip install voxelmorph`
  - Source: `pip install -e .`

## 2. Grab the sample data
The repository already ships with an atlas (`data/atlas.npz`), a labelled test scan (`data/test_scan.npz`), and label definitions (`data/labels.npz`). The `.npz` files contain dense volumes under the `vol` key (and `seg` for segmentations).

```
python - <<'PY'
import numpy as np
for name in ('atlas.npz', 'test_scan.npz'):
    vol = np.load(f'data/{name}')
    print(f'{name}: {vol["vol"].shape}, keys={list(vol.keys())}')
PY
```

## 3. Download a pretrained model
Pick a model from `data/readme.md`. For example, to fetch the dense brain T1 model used in the papers:
```
mkdir -p models
curl -L -o models/dense_brain_T1.h5 \
  https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/models/vxm_dense_brain_T1_3D_mse.h5
```
(If `curl` is unavailable, the file can be downloaded with a browser and saved to `models/`.)

## 4. Run the registration script
```
python scripts/register.py \
    --moving data/test_scan.npz \
    --fixed data/atlas.npz \
    --moved outputs/test_scan_in_atlas_space.nii.gz \
    --warp outputs/test_scan_to_atlas_warp.npz \
    --model models/dense_brain_T1.h5
```

- Create the `outputs/` directory first if it does not exist.
- The script automatically infers the input dimensionality and channel count; add `--multichannel` when your volumes already contain an explicit feature axis.
- The `.npz` deformation field can be reapplied with `scripts/warp.py` or via `voxelmorph.networks.Transform`.

## 5. Inspect the results
Any NIfTI viewer (ITK-SNAP, FSLeyes, Freeview) can visualise the warped scan. To quickly confirm that the file exists and the array shape matches expectations:
```
python - <<'PY'
import nibabel as nib
img = nib.load('outputs/test_scan_in_atlas_space.nii.gz')
print('Shape:', img.shape)
print('Voxel spacing:', img.header.get_zooms())
PY
```

## Next steps
- Try another pair of volumes by replacing the `--moving` and `--fixed` arguments.
- Save the predicted warp (`--warp`) and reuse it with `python scripts/warp.py --warp ...` to transform additional contrasts or segmentations.
- Explore `python scripts/register.py --help` for GPU selection and channel-aware options.
