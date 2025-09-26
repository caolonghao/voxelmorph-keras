# VoxelMorph: learning-based image registration

VoxelMorph is a general-purpose library for learning-based medical image registration, powered by the PyTorch-backed Keras runtime. The project bundles reusable network components, loss functions, data generators, and end-to-end training and inference scripts so you can assemble registration pipelines quickly.

## Highlights
- ready-to-run command-line scripts for training, registering, evaluating, and template building
- modular Keras layers, networks, generators, and losses for custom research workflows
- ships with sample atlases and segmentations plus links to community-pretrained models
- battle-tested on 2D and 3D neuroimaging, but adaptable to any dense deformation task

## Installation

### PyPI (recommended)
```
pip install voxelmorph
```

### From source
Clone the repository and install in editable mode:
```
pip install -e .
```
This reads dependencies from `setup.py`; ensure `torch>=2.0` and `neurite>=0.2` are available.

## Tutorials and guides
- [Quickstart: register two volumes](tutorial/quickstart_registration.md)
- [Quickstart: train a model from scratch](tutorial/quickstart_training.md)
- [Tutorial: build a 3D atlas](tutorial/atlas_generation_3d.md)
- [Tutorial index](tutorial/README.md) for an overview and additional learning resources
- Cloud and notebook experiences: the classic [VoxelMorph tutorial](http://tutorial.voxelmorph.net/) plus the collection of Colab notebooks linked in the tutorial index (SynthMorph, CT-to-MRI, annotation warping, atlas building, and more)

## Quick demo (Python API)
```python
import voxelmorph as vxm

moving = vxm.py.utils.load_volfile('moving.nii.gz', add_batch_axis=True, add_feat_axis=True)
fixed = vxm.py.utils.load_volfile('fixed.nii.gz', add_batch_axis=True, add_feat_axis=True)

model = vxm.networks.VxmDense.load('model.h5', inshape=moving.shape[1:-1])
warp = model.register(moving, fixed)

transform = vxm.networks.Transform(moving.shape[1:-1])
registered = transform.predict([moving, warp])
vxm.py.utils.save_volfile(registered.squeeze(), 'warped.nii.gz')
```
This mirrors the behaviour of the command-line scripts while giving full access to the model objects.

## Working with the command-line scripts
All scripts live in `scripts/` and accept `--help` for the full option set.

### Training
To train a dense deformation model on a list of volumes:
```
python scripts/train.py \
    --img-list /path/to/train_volumes.txt \
    --model-dir /path/to/output \
    --gpu 0
```

- `train_volumes.txt` is a newline-separated list of `.npz`, `.nii.gz`, or `.mgz` files. Use `--img-prefix`/`--img-suffix` if paths need a common prefix or suffix.
- Add `--atlas atlas.npz` to switch to scan-to-atlas training, or `--bidir` to enable bidirectional loss.
- Change loss behaviour with `--image-loss {mse,ncc}` and deformation regularisation with `--lambda`.

### Registration
Given a trained model file (any `voxelmorph.networks.VxmDense` saved via `save()`), register a moving volume to a fixed target:
```
python scripts/register.py \
    --moving moving.npz \
    --fixed atlas.npz \
    --moved moved.nii.gz \
    --model models/brain.h5 \
    --warp moved_warp.npz
```
The script supports `.npz`, `.nii.gz`, and `.mgz` inputs. Use `--multichannel` when images already contain an explicit channel dimension.

### Evaluation (Dice overlap)
Measure Dice scores between warped segmentations and fixed atlases:
```
python scripts/test.py \
    --model models/brain.h5 \
    --pairs /path/to/pairs.txt \
    --img-suffix .npz --seg-suffix .npz \
    --labels data/labels.npz
```
`pairs.txt` contains `moving fixed` pairs per line. Provide matching segmentation prefixes/suffixes to locate label volumes.

### Additional workflows
- `scripts/train_template.py` and `scripts/train_cond_template.py` implement atlas construction
- `scripts/train_hypermorph.py` covers hyperparameter amortisation
- `scripts/train_synthmorph*.py` replicate SynthMorph experiments
- `scripts/warp.py` applies deformations produced by the models to arbitrary volumes

## Pre-trained models and sample data
Sample atlases, probability maps, labels, and quick-start data live under `data/`. The companion `data/readme.md` links to community-pretrained weights, including the dense brain T1 model and SynthMorph variants. Use these assets to test-drive the scripts without sourcing your own data.

## SynthMorph
SynthMorph is a strategy for learning registration without acquired imaging data, producing networks that are agnostic to MRI contrast variations ([eprint arXiv:2004.10282](https://arxiv.org/abs/2004.10282)). For a video overview and interactive demo that synthesises training volumes from label maps, visit [synthmorph.voxelmorph.net](https://synthmorph.voxelmorph.net).

We provide pretrained weights for a ["shapes" variant](https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/shapes-dice-vel-3-res-8-16-32-256f.h5) trained purely on synthetic geometry and a ["brains" variant](https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/brains-dice-vel-0.5-res-16-256f.h5) trained on generated brain labels. The brain model optimises volume-overlap for a curated [set of FreeSurfer structures](https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/fs-labels.npy). Use `python scripts/register.py` with the downloaded weights to run inference.

For best performance:
- min-max normalise inputs so intensities fall in `[0, 1]`
- resample scans into the affine frame of the [reference image](https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/ref.nii.gz)

A common preprocessing route uses FreeSurfer: skull-strip with [SAMSEG](https://surfer.nmr.mgh.harvard.edu/fswiki/Samseg), then align with [`mri_robust_register`](https://surfer.nmr.mgh.harvard.edu/fswiki/mri_robust_register):

```
mri_robust_register --mov in.nii.gz --dst out.nii.gz --lta transform.lta --satit --iscale
mri_robust_register --mov in.nii.gz --dst out.nii.gz --lta transform.lta --satit --iscale --ixform transform.lta --affine
```

Replace `--satit --iscale` with `--cost NMI` when you register across different MRI contrasts.


## Parameter choices

### CVPR version

For the CC loss function, we found a reg parameter of 1 to work best, with more regularisation leading to smoother deformation.

For the segmentation-based loss function, we found that a data weight of 5 to work best, with more regularisation giving higher Dice scores but smoother deformations. However, Dice is already fairly high even for the segmentation-based loss alone.

Note that some of these choices might depend on the dataset and network architecture.

### MICCAI diffeomorphic version

This uses the same architecture as the CVPR version, with the addition of a velocity field integration method (scaling and squaring), paired with a loss on the velocity field. Please see the paper for additional detail. A good range of regularisation parameter is between 0.01 and 0.1. 

### Probabilistic version

This version integrates the probabilistic loss function and uses a sampling decoder for inference. This works best if a variational-type data augmentation is enabled, which introduces random augmentation to each training iteration from a learnt space. Please see the papers for full details.


## Citations

If you use VoxelMorph in your work, please cite the relevant publications.

  * For the probabilistic diffeomorphic model (default torch-backed implementation):

    **Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration**  
[Adrian V. Dalca](http://adalca.mit.edu), [Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
MICCAI 2018. [eprint arXiv:1805.04605](https://arxiv.org/abs/1805.04605)

  * For the original CNN model, MSE, CC, or segmentation-based losses:

    **VoxelMorph: A Learning Framework for Deformable Medical Image Registration**  
[Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [Amy Zhao](http://people.csail.mit.edu/xamyzhao/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
IEEE TMI: Transactions on Medical Imaging. 2019. 
[eprint arXiv:1809.05231](https://arxiv.org/abs/1809.05231)

    **An Unsupervised Learning Model for Deformable Medical Image Registration**  
[Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [Amy Zhao](http://people.csail.mit.edu/xamyzhao/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
CVPR 2018. [eprint arXiv:1802.02604](https://arxiv.org/abs/1802.02604)


## Notes
- **keywords**: machine learning, convolutional neural networks, alignment, mapping, registration  
- **data in papers**: 
In our initial papers, we used publicly available data, but unfortunately we cannot redistribute it (due to the constraints of those datasets). We do a certain amount of pre-processing for the brain images we work with, to eliminate sources of variation and be able to compare algorithms on a level playing field. In particular, we perform FreeSurfer `recon-all` steps up to skull stripping and affine normalization to Talairach space, and crop the images via `((48, 48), (31, 33), (3, 29))`. 

We encourage users to download and process their own data. See [a list of medical imaging datasets here](https://github.com/adalca/medical-datasets). Note that you likely do not need to perform all of the preprocessing steps, and indeed VoxelMorph has been used in other work with other data.


## Creation of deformable templates

To experiment with this method, please use `scripts/train_template.py` for unconditional templates and `scripts/train_cond_template.py` for conditional templates, which use the same conventions as VoxelMorph (please note that these files are less polished than the rest of the VoxelMorph library).

We've also provided an unconditional atlas in `data/generated_uncond_atlas.npz.npy`. 

Models in h5 format weights are provided for [unconditional atlas here](http://people.csail.mit.edu/adalca/voxelmorph/atlas_creation_uncond_NCC_1500.h5), and [conditional atlas here](http://people.csail.mit.edu/adalca/voxelmorph/atlas_creation_cond_NCC_1022.h5).

**Explore the atlases [interactively here](http://voxelmorph.mit.edu/atlas_creation/)** with tipiX!

## Data
While we cannot release most of the datasets used in the original VoxelMorph papers because redistribution is restricted, we thoroughly processed and [re-released OASIS1](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md) while developing [HyperMorph](http://hypermorph.voxelmorph.net/). The accompanying [VoxelMorph-on-OASIS Colab](https://colab.research.google.com/drive/1ZefmWXBupRNsnIbBbGquhVDsk-7R7L1S?usp=sharing) demonstrates an end-to-end training pipeline on that release.

## Contact
For code-related questions please [open an issue](https://github.com/voxelmorph/voxelmorph/issues/new?labels=voxelmorph) or [start a discussion](https://github.com/voxelmorph/voxelmorph/discussions) about general registration topics.
