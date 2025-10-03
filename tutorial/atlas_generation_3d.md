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
python -m voxelmorph.scripts.train_template \
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
python -m voxelmorph.scripts.register \
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

## Optional: supervise atlas building with segmentations
When manual segmentations are available you can encourage anatomical consistency by adding a Dice loss between each subject segmentation warped into atlas space and a learnable atlas segmentation. A convenient way to pair scans and labels is to use a CSV file:

```
image,seg
/data/sub-001_T1w.nii.gz,/data/sub-001_seg.nii.gz
/data/sub-002_T1w.nii.gz,/data/sub-002_seg.nii.gz
...
```

Load the CSV inside a custom training script (for example `scripts/train_template_supervised.py`) that extends `train_template.py`:

```python
import csv

def read_supervised_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        pairs = [(row['image'], row['seg']) for row in reader]
    if not pairs:
        raise ValueError('CSV is empty or missing "image"/"seg" headers.')
    return pairs
```

The model can reuse `TemplateCreation` for the intensity pathway and bolt on a segmentation branch:

```python
template = vxm.networks.TemplateCreation(
    inshape=inshape,
    nb_unet_features=[enc_nf, dec_nf],
    src_feats=nfeats,
    atlas_feats=nfeats,
)

image_input = template.inputs[0]
warped_image, atlas_image, _, flow = template.outputs
seg_input = KL.Input(shape=(*inshape, nb_labels), name='seg_input')

atlas_seg_layer = ne.layers.LocalParamWithInput(
    shape=(*inshape, nb_labels),
    initializer='zeros',
    name='template_creation_atlas_seg_param',
)
atlas_seg = atlas_seg_layer(image_input)

warp_seg = vxm.networks.Transform(inshape, nb_feats=nb_labels, interp_method='nearest')
warped_seg = warp_seg([seg_input, flow])

model = keras.Model(
    inputs=[image_input, seg_input],
    outputs=[warped_image, atlas_image, flow, warped_seg, atlas_seg],
)
```

Expose Dice weights as CLI flags so you can balance segmentation supervision:

```
python -m voxelmorph.scripts.train_template_supervised \
    --data-csv train_pairs.csv \
    --labels-list 1 \
    --dice-loss-weight 1.5 \
    --atlas-dice-loss-weight 0.5 \
    --model-dir outputs/atlas3d_supervised \
    --gpu 0
```

Compile the extended model with an additional Dice loss term, driven by those flags:

```python
image_loss = vxm.losses.NCC().loss
grad_loss = vxm.losses.Grad('l2').loss
dice_loss = vxm.losses.Dice().loss

model.compile(
    optimizer=optimizers.Adam(learning_rate=args.lr),
    loss=[image_loss, vxm.losses.MSE().loss, grad_loss, dice_loss, dice_loss],
    loss_weights=[args.image_loss_weight, args.mean_loss_weight, args.grad_loss_weight, args.dice_loss_weight, args.atlas_dice_loss_weight],
)
```

During training the generator should emit two inputs (`[image, one_hot_seg]`) and two segmentation targets (the subject segmentation warped into atlas space and the learnable atlas segmentation). You can follow the implementation of `vxm.generators.semisupervised` for converting label maps into one-hot volumes and down-sampling them if needed.

Use `labels.npy` to list the integer label values that should contribute to the Dice loss (for example `[0, 1, 2, 3]`). The learned atlas segmentation can be exported alongside the intensity atlas and reused to propagate annotations via `scripts/warp.py` or the Python API.
