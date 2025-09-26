# Quickstart: train a model from scratch

This guide shows how to launch an unsupervised VoxelMorph training run using the bundled command-line tools. The default configuration matches the probabilistic diffeomorphic model introduced in the MICCAI 2018 paper.

## 1. Requirements
- Python 3.9+
- `torch>=2.0` with GPU support recommended but not mandatory
- `neurite>=0.2` (installed automatically by `voxelmorph`)
- Dataset stored as `.npz`, `.nii.gz`, or `.mgz` volumes with consistent shapes

Install VoxelMorph:
```
pip install voxelmorph
```
Or, from the repository checkout:
```
pip install -e .
```

## 2. Organise your training set
Create a text file containing one absolute or relative path per line. For example:
```
# train_volumes.txt
/path/to/sub-001_T1w.npz
/path/to/sub-002_T1w.npz
...
```
When using `.npz` files, ensure a `vol` array is present (and optionally `seg` for semi-supervised workflows). The helper below validates the list:
```
python - <<'PY'
from voxelmorph.py import utils
paths = utils.read_file_list('train_volumes.txt')
print(f'Loaded {len(paths)} volumes')
PY
```
If you just want to smoke-test the pipeline, duplicate the bundled `data/test_scan.npz` a few times. Real training requires diverse subjects.

## 3. (Optional) Prepare an atlas
For scan-to-atlas training, provide an `.npz` file with `vol` (and optionally `seg`). The repository ships a sample brain atlas at `data/atlas.npz`.

## 4. Launch training
```
python scripts/train.py \
    --img-list train_volumes.txt \
    --model-dir outputs/models \
    --gpu 0 \
    --epochs 50 \
    --steps-per-epoch 100
```
Key flags:
- `--atlas data/atlas.npz` trains scan-to-atlas instead of scan-to-scan
- `--int-steps` controls diffeomorphic integration depth (set to `0` for the original non-diffeomorphic network)
- `--image-loss {mse,ncc}` switches the image similarity term
- `--lambda` adjusts deformation smoothness; lower values allow more flexible warps
- `--use-probs` enables the probabilistic velocity field formulation

The script prints progress every epoch and writes checkpoints (Keras `.h5` files) plus the training configuration into `--model-dir`.

## 5. Resume or fine-tune
To continue training from an existing checkpoint:
```
python scripts/train.py \
    --img-list train_volumes.txt \
    --model-dir outputs/models \
    --load-weights outputs/models/weights.050.h5 \
    --initial-epoch 50
```
Adjust filenames to match your saved checkpoints.

## 6. Evaluate the model
Once training stabilises, compute Dice scores on held-out subjects with `scripts/test.py`:
```
python scripts/test.py \
    --model outputs/models/weights.latest.h5 \
    --pairs val_pairs.txt \
    --img-suffix .npz \
    --seg-suffix .npz \
    --labels data/labels.npz
```
Required inputs:
- `val_pairs.txt`: whitespace-separated `moving fixed` pairs
- Segmentation volumes containing a `seg` array aligned with each subject
- Optional `--labels` `.npy` file listing numeric labels for Dice computation

## 7. Deploy the checkpoint
The `.h5` weights saved in `--model-dir` are compatible with `scripts/register.py` and the Python API (`voxelmorph.networks.VxmDense.load`). Ship them with your inference script or convert to other formats if needed.

## Troubleshooting
- **Out-of-memory**: lower `--batch-size`, reduce `--int-steps`, or crop inputs.
- **Misaligned results**: verify that all training volumes share the same voxel spacing and shape; resample beforehand.
- **Divergent loss**: start with `--image-loss ncc` for intensity-normalised T1 MRI, or clip intensities for multi-modal data.

For deeper control, inspect the script source in `scripts/train.py` and the modules under `voxelmorph/`.
