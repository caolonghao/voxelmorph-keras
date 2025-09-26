# Tutorial: build a 3D atlas with VoxelMorph

This guide walks through the unconditional atlas workflow backed by the PyTorch+Keras Voxelmorph stack. The training script (`scripts/train_template.py`) optimises a learnable 3D template together with deformation fields that align each subject to the template.

## 1. Requirements
- Python 3.9+
- `torch>=2.0`, `neurite>=0.2`, and `voxelmorph>=0.2` installed (either via `pip install voxelmorph` or `pip install -e .` inside the repo)
- Training data in `.npz`, `.nii.gz`, or `.mgz` format with consistent voxel spacing and shape
- Optional GPU for faster convergence (set with `--gpu`)

## 2. Assemble a subject list
Create a newline-separated text file pointing to every 3D volume you want to include. Paths can be absolute or relative.

```
# train_scans.txt
/path/to/sub-001_T1w.nii.gz
/path/to/sub-002_T1w.nii.gz
...
```

Sanity-check that Voxelmorph can read the list and that shapes match:
```
python - <<'PY'
from voxelmorph.py import utils
import nibabel as nib
paths = utils.read_file_list('train_scans.txt')
print(f'Loaded {len(paths)} paths')
img = nib.load(paths[0])
print('Example shape:', img.shape)
PY
```

When using `.npz` files, ensure each file contains a `vol` array (and optionally `seg`).

## 3. (Optional) Provide a seed atlas
Passing `--init-template some_atlas.nii.gz` seeds the learnable template. If you omit it, the script averages the first 100 scans to generate an initial atlas and saves it as `init_template.nii.gz` in the output directory.

## 4. Launch atlas training
```
python scripts/train_template.py \
    --img-list train_scans.txt \
    --model-dir outputs/atlas3d \
    --gpu 0 \
    --epochs 300 \
    --steps-per-epoch 100 \
    --batch-size 1
```

Key flags:
- `--image-loss {ncc,mse}` switches the similarity term (default `ncc`)
- `--image-loss-weight`, `--mean-loss-weight`, `--grad-loss-weight` tune the balance between reconstruction, mean-alignment, and deformation smoothness
- `--enc` / `--dec` supply custom U-Net filter layouts, e.g. `--enc 16 32 64 64 --dec 64 64 64 32 16`
- `--multichannel` should be set when inputs already contain an explicit channel dimension

During training `ModelCheckpoint` writes weights named `weights_epoch_XXXX.weights.h5`. The atlas parameter lives inside the `TemplateCreation` network as a `neurite.layers.LocalParamWithInput`, so the script snapshots monophasic weights even if training is interrupted.

## 5. Outputs
After training completes, the script saves:
- `template.nii.gz`: the optimised 3D atlas volume
- `init_template.nii.gz`: the initial average (unless an explicit seed was supplied)
- `weights_epoch_*.weights.h5`: model checkpoints compatible with `voxelmorph.networks.TemplateCreation.load`

Inspect the atlas with any NIfTI viewer or via Python:
```
python - <<'PY'
import nibabel as nib
atlas = nib.load('outputs/atlas3d/template.nii.gz')
print('Atlas shape:', atlas.shape)
print('Voxel spacing:', atlas.header.get_zooms())
PY
```

## 6. Register subjects into atlas space
Use the saved atlas weights with the standard registration script:
```
python scripts/register.py \
    --moving some_subject.nii.gz \
    --fixed outputs/atlas3d/template.nii.gz \
    --moved outputs/atlas3d/some_subject_in_atlas.nii.gz \
    --model outputs/atlas3d/weights_epoch_0300.weights.h5
```

The registration network reuses the same deformation blocks provided by the `TemplateCreation` model, so the interface is identical to atlas-free registration.

## 7. Extending or conditioning the atlas
- To incorporate phenotype metadata (age, diagnosis, etc.), switch to `scripts/train_cond_template.py`, which wraps `voxelmorph.networks.ConditionalTemplateCreation` and expects `--pheno-csv`.
- For alternative integration depths or probabilistic flows, pass `--int-steps` or `--use-probs` (the same arguments accepted by `VxmDense`).

## Troubleshooting
- **Divergent loss**: ensure all inputs share the same field-of-view; add pre-alignment if necessary.
- **Blurry atlas**: try increasing `--image-loss-weight` or providing a higher-quality seed atlas.
- **Overly sharp deformations**: raise `--grad-loss-weight` or limit the decoder depth via `--enc/--dec`.

Interface compatibility check: both `scripts/train_template.py` and `voxelmorph.networks.TemplateCreation` rely on the current `neurite.layers.LocalParamWithInput` and `voxelmorph.networks.VxmDense` modules, so the commands above match the latest library layout.
