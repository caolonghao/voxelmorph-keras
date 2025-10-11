#!/usr/bin/env python

"""
Template creation script with supervised segmentation guidance.

This script extends the standard template training workflow by consuming a CSV
file that lists image and segmentation-mask pairs. The template network is
encouraged to match both the subject intensities and their corresponding
segmentation masks once the atlas parameters are warped into subject space.

Example CSV format (header required):

    image,mask
    /path/to/subj_01_image.nii.gz,/path/to/subj_01_mask.nii.gz
    /path/to/subj_02_image.nii.gz,/path/to/subj_02_mask.nii.gz

If you use this code, please cite:

    Learning Conditional Deformable Templates with Convolutional Networks
    Adrian V. Dalca, Marianne Rakic, John Guttag, Mert R. Sabuncu
    NeurIPS 2019. https://arxiv.org/abs/1908.02738
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import argparse
import csv
from typing import Iterable, List, Tuple

import numpy as np
import keras

import voxelmorph as vxm
from voxelmorph import keras_backend as tf


def _read_pairs_from_csv(
    csv_path: str,
    image_column: str,
    mask_column: str,
    delimiter: str,
    img_prefix: str = "",
    img_suffix: str = "",
    mask_prefix: str = "",
    mask_suffix: str = "",
) -> List[Tuple[str, str]]:
    """
    Read (image, mask) pairs from a CSV file.

    Relative paths are resolved with respect to the CSV location.
    """
    pairs: List[Tuple[str, str]] = []
    csv_dir = os.path.dirname(os.path.abspath(csv_path))

    with open(csv_path, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        if image_column not in reader.fieldnames or mask_column not in reader.fieldnames:
            raise ValueError(
                f'CSV file must contain columns "{image_column}" and "{mask_column}". '
                f'Available columns: {reader.fieldnames}'
            )

        for row in reader:
            img_entry = row[image_column].strip()
            mask_entry = row[mask_column].strip()
            if not img_entry or not mask_entry:
                continue

            img_path = img_entry if os.path.isabs(img_entry) else os.path.join(csv_dir, img_entry)
            mask_path = mask_entry if os.path.isabs(mask_entry) else os.path.join(csv_dir, mask_entry)

            img_path = os.path.join(img_prefix, img_path) if img_prefix else img_path
            mask_path = os.path.join(mask_prefix, mask_path) if mask_prefix else mask_path

            if img_suffix:
                img_path = f'{img_path}{img_suffix}'
            if mask_suffix:
                mask_path = f'{mask_path}{mask_suffix}'

            if not os.path.isfile(img_path):
                raise FileNotFoundError(f'Image file not found: {img_path}')
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f'Mask file not found: {mask_path}')

            pairs.append((img_path, mask_path))

    if not pairs:
        raise ValueError('No valid (image, mask) pairs were found in the CSV file.')
    return pairs


def _load_average_template(
    pairs: Iterable[Tuple[str, str]],
    num_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the average image and mask templates from the first num_samples pairs.

    Returns:
        image_template: (1, *shape, 1)
        mask_template: (1, *shape, 1)
    """
    image_sum = None
    mask_sum = None

    for idx, (img_path, mask_path) in enumerate(pairs):
        if idx >= num_samples:
            break

        img = vxm.py.utils.load_volfile(
            img_path, add_batch_axis=True, add_feat_axis=True
        ).astype(np.float32)
        mask = vxm.py.utils.load_volfile(
            mask_path, add_batch_axis=True, add_feat_axis=True
        ).astype(np.float32)

        if img.shape != mask.shape:
            raise ValueError(f'Image and mask shapes must match. Got {img.shape} vs {mask.shape}.')

        image_sum = img if image_sum is None else image_sum + img
        mask_sum = mask if mask_sum is None else mask_sum + mask

    if image_sum is None or mask_sum is None:
        raise ValueError('Unable to compute template averages (empty dataset).')

    divisor = float(num_samples)
    return image_sum / divisor, mask_sum / divisor


def _supervised_template_generator(
    pairs: List[Tuple[str, str]],
    batch_size: int,
) -> Iterable[Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Yield batches for supervised template training.

    Inputs:
        combined volume with two channels [image, mask].
    Targets:
        warped image target, warped mask target, zeros (for inverse atlas),
        zeros (mean stream), zeros (gradient regularization).
    """
    zeros_scalar = None
    zeros_vector = None

    while True:
        indices = np.random.choice(len(pairs), size=batch_size, replace=len(pairs) < batch_size)

        img_batch = []
        mask_batch = []
        for idx in indices:
            img_path, mask_path = pairs[idx]
            img = vxm.py.utils.load_volfile(
                img_path, add_batch_axis=True, add_feat_axis=True
            ).astype(np.float32)
            mask = vxm.py.utils.load_volfile(
                mask_path, add_batch_axis=True, add_feat_axis=True
            ).astype(np.float32)

            if img.shape != mask.shape:
                raise ValueError(
                    f'Image and mask shapes must match. Got {img.shape} vs {mask.shape} for '
                    f'{img_path} / {mask_path}.'
                )

            img_batch.append(img)
            mask_batch.append(mask)

        img_batch = np.concatenate(img_batch, axis=0)
        mask_batch = np.concatenate(mask_batch, axis=0)
        combined = np.concatenate([img_batch, mask_batch], axis=-1)

        if zeros_scalar is None:
            spatial_shape = img_batch.shape[1:-1]
            zeros_scalar = np.zeros((batch_size, *spatial_shape, 1), dtype=np.float32)
            zeros_vector = np.zeros((batch_size, *spatial_shape, len(spatial_shape)), dtype=np.float32)

        yield (
            [combined],
            [img_batch, mask_batch, zeros_scalar, zeros_scalar, zeros_vector],
        )


def _build_supervised_template_model(
    inshape: Tuple[int, ...],
    enc_nf: List[int],
    dec_nf: List[int],
    image_loss_name: str,
    seg_interp: str,
    image_loss_weight: float,
    seg_loss_weight: float,
    neg_loss_weight: float,
    mean_loss_weight: float,
    grad_loss_weight: float,
) -> Tuple[keras.Model, List, List, vxm.networks.TemplateCreation]:
    """
    Construct the supervised template model and associated losses/weights.
    """
    base_model = vxm.networks.TemplateCreation(
        inshape,
        nb_unet_features=[enc_nf, dec_nf],
        atlas_feats=2,
        src_feats=2,
    )

    # Base outputs
    warped_full = base_model.outputs[0]
    neg_full = base_model.outputs[1]
    mean_stream = base_model.outputs[2]
    pos_flow = base_model.outputs[3]

    # Image channel (linear interpolation provided by base model)
    warped_img = keras.layers.Lambda(lambda x: x[..., :1], name='warped_image')(warped_full)
    neg_img = keras.layers.Lambda(lambda x: x[..., :1], name='neg_image')(neg_full)

    # Segmentation atlas warped with nearest-neighbour interpolation
    atlas_params = base_model.references.atlas_layer(base_model.inputs[0])
    atlas_seg = keras.layers.Lambda(lambda x: x[..., 1:2], name='atlas_seg')(atlas_params)
    warped_seg = vxm.layers.SpatialTransformer(
        interp_method=seg_interp,
        name='warped_segmentation'
    )([atlas_seg, pos_flow])

    model = keras.Model(
        inputs=base_model.inputs,
        outputs=[warped_img, warped_seg, neg_img, mean_stream, pos_flow],
        name='template_creation_supervised'
    )
    # Expose references for downstream usage (set/get atlas, flows, etc.)
    model.references = base_model.references

    # Image reconstruction loss
    if image_loss_name == 'ncc':
        image_loss_fn = vxm.losses.NCC().loss
    elif image_loss_name == 'mse':
        image_loss_fn = vxm.losses.MSE().loss
    elif image_loss_name == 'ssim':
        image_loss_fn = vxm.losses.SSIM().loss
    else:
        raise ValueError(f'Image loss should be "mse", "ncc", or "ssim", but found "{image_loss_name}".')

    dice_loss_fn = vxm.losses.Dice().loss

    def neg_loss_fn(_, y_pred):
        atlas_batch = model.references.atlas_layer(y_pred)
        atlas_img = atlas_batch[..., :1]
        return image_loss_fn(atlas_img, y_pred)

    losses = [
        image_loss_fn,
        dice_loss_fn,
        neg_loss_fn,
        vxm.losses.MSE().loss,
        vxm.losses.Grad('l2', loss_mult=2).loss,
    ]

    loss_weights = [
        image_loss_weight,
        seg_loss_weight,
        neg_loss_weight,
        mean_loss_weight,
        grad_loss_weight,
    ]

    return model, losses, loss_weights, base_model


def main():
    parser = argparse.ArgumentParser(description='Supervised template creation with segmentation guidance.')

    # CSV / dataset parameters
    parser.add_argument('--pairs-csv', required=True, help='CSV listing image/mask pairs.')
    parser.add_argument('--image-column', default='image', help='CSV column containing image paths.')
    parser.add_argument('--mask-column', default='mask', help='CSV column containing mask paths.')
    parser.add_argument('--delimiter', default=',', help='CSV delimiter (default: ",").')
    parser.add_argument('--img-prefix', default='', help='Optional prefix prepended to image paths.')
    parser.add_argument('--img-suffix', default='', help='Optional suffix appended to image paths.')
    parser.add_argument('--mask-prefix', default='', help='Optional prefix prepended to mask paths.')
    parser.add_argument('--mask-suffix', default='', help='Optional suffix appended to mask paths.')

    # Training parameters
    parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='number of steps per epoch (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--load-weights', help='optional weights file to initialize with')
    parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
    parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')
    parser.add_argument('--checkpoint-frequency', type=int, default=10,
                        help='number of epochs between checkpoints (default: 10)')

    # Optional initial template
    parser.add_argument('--init-image-template', help='Optional initial atlas image volume.')
    parser.add_argument('--init-mask-template', help='Optional initial atlas mask volume.')

    # Network architecture parameters
    parser.add_argument('--enc', type=int, nargs='+', help='list of Unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+', help='list of Unet decoder filters (default: 32 32 32 32 32 16 16)')

    # Loss hyperparameters
    parser.add_argument('--image-loss', default='ncc', help='image reconstruction loss: mse, ncc, or ssim (default: ncc)')
    parser.add_argument('--image-loss-weight', type=float, default=1.0, help='weight for image reconstruction loss')
    parser.add_argument('--seg-loss-weight', type=float, default=1.0, help='weight for segmentation loss')
    parser.add_argument('--neg-loss-weight', type=float, default=None,
                        help='weight for atlas regularization loss (default: 1 - image_loss_weight)')
    parser.add_argument('--mean-loss-weight', type=float, default=1.0, help='weight for mean stream loss')
    parser.add_argument('--grad-loss-weight', type=float, default=1.0, help='weight for flow gradient loss')
    parser.add_argument('--seg-interp', default='nearest', choices=['nearest', 'linear'],
                        help='interpolation used when warping the segmentation atlas (default: nearest)')

    args = parser.parse_args()

    pairs = _read_pairs_from_csv(
        args.pairs_csv,
        args.image_column,
        args.mask_column,
        args.delimiter,
        img_prefix=args.img_prefix,
        img_suffix=args.img_suffix,
        mask_prefix=args.mask_prefix,
        mask_suffix=args.mask_suffix,
    )

    os.makedirs(args.model_dir, exist_ok=True)

    # Determine template shape from first subject
    first_img = vxm.py.utils.load_volfile(
        pairs[0][0], add_batch_axis=True, add_feat_axis=True
    ).astype(np.float32)
    spatial_shape = first_img.shape[1:-1]

    # Initial template
    if args.init_image_template and args.init_mask_template:
        template_img = vxm.py.utils.load_volfile(
            args.init_image_template, add_batch_axis=True, add_feat_axis=True
        ).astype(np.float32)
        template_mask = vxm.py.utils.load_volfile(
            args.init_mask_template, add_batch_axis=True, add_feat_axis=True
        ).astype(np.float32)
        if template_img.shape != template_mask.shape:
            raise ValueError('Initial image template and mask template must have matching shapes.')
        template = np.concatenate([template_img, template_mask], axis=-1)
    elif args.init_image_template or args.init_mask_template:
        raise ValueError('Provide both --init-image-template and --init-mask-template, or neither.')
    else:
        navgs = min(100, len(pairs))
        print(f'Creating starting templates by averaging first {navgs} subjects.')
        template_img, template_mask = _load_average_template(pairs, navgs)
        template = np.concatenate([template_img, template_mask], axis=-1)

    vxm.py.utils.save_volfile(
        template[..., 0].squeeze(),
        os.path.join(args.model_dir, 'init_template_image.nii.gz')
    )
    vxm.py.utils.save_volfile(
        template[..., 1].squeeze(),
        os.path.join(args.model_dir, 'init_template_mask.nii.gz')
    )

    device, nb_devices = vxm.utils.setup_device(args.gpu)
    if nb_devices > 1:
        raise NotImplementedError('Multi-GPU training is not implemented in this backend build.')

    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    neg_loss_weight = args.neg_loss_weight if args.neg_loss_weight is not None else max(1.0 - args.image_loss_weight, 0.0)

    model, losses, loss_weights, base_model = _build_supervised_template_model(
        inshape=spatial_shape,
        enc_nf=enc_nf,
        dec_nf=dec_nf,
        image_loss_name=args.image_loss,
        seg_interp=args.seg_interp,
        image_loss_weight=args.image_loss_weight,
        seg_loss_weight=args.seg_loss_weight,
        neg_loss_weight=neg_loss_weight,
        mean_loss_weight=args.mean_loss_weight,
        grad_loss_weight=args.grad_loss_weight,
    )

    base_model.set_atlas(template)

    if args.load_weights:
        model.load_weights(args.load_weights, by_name=True)

    checkpoint_epochs = max(args.checkpoint_frequency, 1)
    checkpoint_steps = checkpoint_epochs * args.steps_per_epoch
    save_callback = keras.callbacks.ModelCheckpoint(
        os.path.join(args.model_dir, '{epoch:04d}.keras'),
        save_freq=checkpoint_steps
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=losses,
        loss_weights=loss_weights
    )

    model.save(os.path.join(args.model_dir, f'{args.initial_epoch:04d}.keras'))

    generator = _supervised_template_generator(pairs, batch_size=args.batch_size)

    model.fit(
        generator,
        initial_epoch=args.initial_epoch,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=[save_callback],
        verbose=1,
    )

    final_atlas = base_model.get_atlas()
    vxm.py.utils.save_volfile(
        final_atlas[..., 0].squeeze(),
        os.path.join(args.model_dir, 'template_image.nii.gz')
    )
    vxm.py.utils.save_volfile(
        final_atlas[..., 1].squeeze(),
        os.path.join(args.model_dir, 'template_mask.nii.gz')
    )


if __name__ == '__main__':
    main()
